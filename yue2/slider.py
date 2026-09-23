"""Routed AR adapters for YuE2.

Each projection owns a low-rank pair plus a bridge from ``stamp.bridge()``.
The bridge class is not reimplemented in this repo. Scale 0 skips the adapter
and returns the frozen projection exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from yue2.backend import architecture_of, attention_targets
from yue2.defaults import FORMATS, PRODUCT_FORMAT, UPSTREAM_REVISION
from yue2.prompts import sound_only


class RoutedProjection(nn.Module):
    def __init__(self, name: str, module: nn.Linear, particles: nn.Parameter, stamp):
        super().__init__()
        rank = int(stamp.spec["adapter_rank"])
        alpha = float(stamp.spec["adapter_alpha"])
        if rank < 1 or alpha <= 0:
            raise ValueError("stamp rank and alpha must be positive")
        self.lora_name = name
        self.rank = rank
        self.scale = alpha / rank
        self.lora_down = nn.Linear(module.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, module.out_features, bias=False)
        self.register_buffer("alpha", torch.tensor(alpha))
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)
        self.bridge = stamp.bridge()
        self.multiplier = 0.0
        object.__setattr__(self, "_particles", particles)
        self.org_forward = module.forward
        module.forward = self.forward

    def forward(self, x):
        base = self.org_forward(x)
        if self.multiplier == 0:
            return base
        features = self.lora_down(x.to(device=self.lora_down.weight.device, dtype=torch.float32))
        routed = self.bridge(features, self._particles)
        delta = self.lora_up(routed).to(device=x.device, dtype=x.dtype)
        return base + delta * (self.multiplier * self.scale)


class ParticleSlider(nn.Module):
    def __init__(self, model, stamp):
        super().__init__()
        targets = attention_targets(model)
        if any(isinstance(getattr(module.forward, "__self__", None), RoutedProjection) for module in targets.values()):
            raise ValueError("A YuE2 slider is already attached")
        self.stamp_id = f"{stamp.architecture_id}/{stamp.formulation_id}"
        self.rank = int(stamp.spec["adapter_rank"])
        self.alpha = float(stamp.spec["adapter_alpha"])
        self.parts = int(stamp.spec["parts"])
        self.particle_dim = int(stamp.spec["particle_dim"])
        self.architecture = architecture_of(model)
        self.target_names = list(targets)
        model.requires_grad_(False)
        device = next(model.parameters()).device
        self.particles = nn.Parameter(torch.randn(self.parts, self.particle_dim, device=device))
        self.adapters = nn.ModuleDict()
        for path, module in targets.items():
            key = path.replace(".", "-")
            self.adapters[key] = RoutedProjection(key, module, self.particles, stamp)
        self.to(device=device, dtype=torch.float32)

    def scaled(self, scale: float):
        return _Scale(self, scale)

    def metadata(self, extra: dict) -> dict:
        record = dict(extra)
        record.update(
            format=PRODUCT_FORMAT,
            rank=self.rank,
            alpha=self.alpha,
            particles=self.parts,
            particle_dim=self.particle_dim,
            architecture=self.architecture,
            targets=self.target_names,
            upstream_revision=UPSTREAM_REVISION,
            formulation=self.stamp_id,
        )
        return record

    def export_state(self, state=None) -> dict[str, torch.Tensor]:
        source = self.state_dict() if state is None else state
        exported = {key: value.detach().float().cpu().contiguous() for key, value in source.items()}
        if any(not torch.isfinite(tensor).all() for tensor in exported.values()):
            raise ValueError("Refusing a non-finite YuE2 slider export")
        return exported

    def save(self, path, metadata: dict, *, state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = self.metadata(metadata)
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
        sound_only(payload)
        tensors = self.export_state(state)
        if set(tensors) != set(self.state_dict()):
            raise ValueError("Export state does not match this slider")
        save_file(tensors, str(path), metadata={"yue2_particle_sliders": payload})
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, model, path, stamp, *, allow_dummy: bool = False):
        path = Path(path)
        record = read_record(path)
        if record.get("format") not in FORMATS:
            raise ValueError("Not a YuE2 routed-particle slider checkpoint")
        if record.get("dummy") and not allow_dummy:
            raise ValueError("Dummy checkpoints cannot be used for song generation")
        if record.get("architecture") != architecture_of(model):
            raise ValueError("Slider architecture does not match this YuE2 model")
        if record.get("targets") != list(attention_targets(model)):
            raise ValueError("Slider target list does not match YuE2 AR attention")
        state = load_file(str(path), device="cpu")
        expected = _expected_shapes(model, stamp)
        if set(state) != set(expected) or any(tuple(state[key].shape) != shape for key, shape in expected.items()):
            raise ValueError("Incomplete or incompatible YuE2 slider tensors")
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("Non-finite YuE2 slider weights")
        alpha = float(stamp.spec["adapter_alpha"])
        if any(float(state[key]) != alpha for key in state if key.endswith(".alpha")):
            raise ValueError("Slider alpha does not match the winning formulation")
        network = cls(model, stamp)
        network.load_state_dict(state, strict=True)
        return network, record


class _Scale:
    def __init__(self, network: ParticleSlider, scale: float):
        if not math.isfinite(scale):
            raise ValueError("Slider scale must be finite")
        self.network = network
        self.scale = float(scale)
        self.previous: list[float] = []

    def __enter__(self):
        self.previous = [adapter.multiplier for adapter in self.network.adapters.values()]
        for adapter in self.network.adapters.values():
            adapter.multiplier = self.scale
        return self.network

    def __exit__(self, exc_type, exc, tb):
        for adapter, value in zip(self.network.adapters.values(), self.previous):
            adapter.multiplier = value
        return False


def read_record(path) -> dict:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        payload = metadata.get("yue2_particle_sliders") or metadata.get("conceptmod") or "{}"
    record = json.loads(payload)
    sound_only(json.dumps(record, ensure_ascii=False))
    if not isinstance(record, dict) or not record:
        raise ValueError("YuE2 slider checkpoint is missing its metadata")
    return record


def _expected_shapes(model, stamp) -> dict[str, tuple[int, ...]]:
    """Shapes from the live stamp bridge, not a copied MLP width table."""
    probe = stamp.bridge().state_dict()
    rank = int(stamp.spec["adapter_rank"])
    expected: dict[str, tuple[int, ...]] = {
        "particles": (int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]))
    }
    for name, module in attention_targets(model).items():
        prefix = "adapters." + name.replace(".", "-")
        expected[prefix + ".lora_down.weight"] = (rank, module.in_features)
        expected[prefix + ".lora_up.weight"] = (module.out_features, rank)
        expected[prefix + ".alpha"] = ()
        for key, tensor in probe.items():
            expected[f"{prefix}.bridge.{key}"] = tuple(tensor.shape)
    return expected
