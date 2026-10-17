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
from yue2.defaults import CORE_PIN, FORMATS, MODEL_ID, PRODUCT_FORMAT, UPSTREAM_REVISION
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
    def load(cls, model, path, stamp, *, allow_dummy: bool = False, model_identity=None):
        # All validation precedes the constructor, which installs live hooks.
        record, state = validate_checkpoint(
            path, stamp, model=model, allow_dummy=allow_dummy, model_identity=model_identity
        )
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


def validate_record(record, stamp, *, allow_dummy=False, model_identity=None):
    """Validate current provenance and the explicitly supported published format."""
    if record.get("format") not in FORMATS:
        raise ValueError("Not a YuE2 routed-particle slider checkpoint")
    if record.get("dummy") and not allow_dummy:
        raise ValueError("Dummy checkpoints cannot be used for song generation")
    current = record["format"] == PRODUCT_FORMAT
    required = {
        "rank": int(stamp.spec["adapter_rank"]),
        "alpha": float(stamp.spec["adapter_alpha"]),
        "particles": int(stamp.spec["parts"]),
        "particle_dim": int(stamp.spec["particle_dim"]),
        "upstream_revision": UPSTREAM_REVISION,
        "model_id": MODEL_ID,
    }
    # Published conceptmod exports named the architecture recipe `recipe` and
    # predate formulation/core identifiers. Keep their recorded provenance.
    recipe = record.get("recipe_name", record.get("recipe"))
    if recipe != stamp.spec["recipe_name"]:
        raise ValueError("Slider recipe does not match the winning formulation")
    if current:
        required.update(
            core_pin=CORE_PIN,
            architecture_id=stamp.architecture_id,
            formulation_id=stamp.formulation_id,
            formulation=f"{stamp.architecture_id}/{stamp.formulation_id}",
        )
    else:
        for key, value in {
            "core_pin": CORE_PIN,
            "architecture_id": stamp.architecture_id,
            "formulation_id": stamp.formulation_id,
            "formulation": f"{stamp.architecture_id}/{stamp.formulation_id}",
        }.items():
            if key in record:
                required[key] = value
        for key, spec_key in (("bridge_width", "adapter_width"), ("router_width", "router_width")):
            required[key] = int(stamp.spec[spec_key])
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"Slider {key} does not match the winning formulation/model")
    identity = record.get("model_identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Slider is missing its base model_identity")
    if model_identity is not None and identity != model_identity:
        raise ValueError("Slider was trained on different base model weights")
    if bool(identity.get("dummy")) != bool(record.get("dummy")):
        raise ValueError("Slider model_identity and dummy provenance disagree")
    if record.get("training_game") == "conditional-pg09-v1":
        _validate_game_signature(record)
    elif "training_game" in record:
        raise ValueError("Unknown slider training_game")
    return record


def _validate_game_signature(record):
    from yue2.integration import LOCK_PATH, PG_VERSION

    signature = record.get("game_signature")
    if not isinstance(signature, dict) or not signature:
        raise ValueError("Conditional PG09 slider is missing its game_signature")
    expected = {
        "pg_version": PG_VERSION, "objective": "g_adv+public_prior;d_adv+public_penalty",
        "response_domain": "continuous_vector", "generator_arm": "routed_particle_gan",
    }
    if any(signature.get(key) != value for key, value in expected.items()):
        raise ValueError("Conditional PG09 version, objective or response signature differs")
    backend = signature.get("backend")
    if not isinstance(backend, dict) or not isinstance(backend.get("model_identity"), dict):
        raise ValueError("Conditional PG09 backend signature is missing")
    binding = backend["model_identity"]
    if binding.get("base") != record["model_identity"]:
        raise ValueError("Conditional PG09 base identity differs from its signature")
    dependencies = record.get("integration_dependencies")
    if not isinstance(dependencies, dict) or binding.get("integration_dependencies") != dependencies:
        raise ValueError("Conditional PG09 integration source signature differs")
    lock = json.loads(LOCK_PATH.read_text())
    core = dependencies.get("core")
    pg = dependencies.get("particlegan")
    if (not isinstance(core, dict) or core.get("commit") != CORE_PIN or not core.get("sources")
            or not isinstance(pg, dict) or pg != lock["particlegan"]
            or any(core.get(key) != value for key, value in lock["core"].items())):
        raise ValueError("Conditional PG09 installed package lineage differs")
    if (backend.get("family") != "yue2_native_ar" or backend.get("student_conditioning") != "neutral"
            or backend.get("trained_scales") != [1.0]):
        raise ValueError("Conditional PG09 conditioning or trained scales differ")
    if record.get("training_weights") != "active" or record.get("ema") is not False:
        raise ValueError("Conditional PG09 inference requires the active training weights")
    recipe = signature.get("recipe")
    if (not isinstance(recipe, dict) or recipe.get("name") != record.get("training_recipe")
            or recipe.get("training_precision") != "float32"):
        raise ValueError("Conditional PG09 training recipe differs from its signature")


def validate_checkpoint(path, stamp, *, model=None, allow_dummy=False, model_identity=None):
    """Read and validate the same tensor payload for the CLI and Comfy gate."""
    record = validate_record(
        read_record(path), stamp, allow_dummy=allow_dummy, model_identity=model_identity
    )
    if model is not None:
        if not record.get("dummy") and model_identity is None:
            raise ValueError("Loading a live slider requires the actual base model_identity")
        if record.get("architecture") != architecture_of(model):
            raise ValueError("Slider architecture does not match this YuE2 model")
        if record.get("targets") != list(attention_targets(model)):
            raise ValueError("Slider target list does not match YuE2 AR attention")
        expected = _expected_shapes(model, stamp)
    else:
        expected = _record_shapes(record, stamp)
    state = load_file(str(path), device="cpu")
    if set(state) != set(expected) or any(tuple(state[key].shape) != shape for key, shape in expected.items()):
        raise ValueError("Incomplete or incompatible YuE2 slider tensors")
    if any(not tensor.is_floating_point() or not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Non-finite or non-floating YuE2 slider weights")
    alpha = float(stamp.spec["adapter_alpha"])
    if any(float(state[key]) != alpha for key in state if key.endswith(".alpha")):
        raise ValueError("Slider alpha does not match the winning formulation")
    return record, state


def _record_shapes(record, stamp):
    """Native AR shapes without constructing or loading a base model."""
    architecture = record.get("architecture")
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")
    if not isinstance(architecture, dict) or any(
        type(architecture.get(key)) is not int or architecture[key] < 1 for key in keys
    ):
        raise ValueError("Slider is missing a valid YuE2 architecture")
    hidden, layers, heads, kv_heads, head_dim = (architecture[key] for key in keys)
    if hidden != heads * head_dim or heads % kv_heads:
        raise ValueError("Slider YuE2 attention dimensions disagree")
    targets = [f"model.layers.{index}.self_attn.{name}"
               for index in range(layers) for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
    if record.get("targets") != targets:
        raise ValueError("Slider target list does not match YuE2 AR attention")
    rank = int(stamp.spec["adapter_rank"])
    bridge = _bridge_shapes(stamp)
    expected = {"particles": (int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]))}
    for path in targets:
        prefix = "adapters." + path.replace(".", "-")
        output = kv_heads * head_dim if path.endswith(("k_proj", "v_proj")) else hidden
        expected[prefix + ".lora_down.weight"] = (rank, hidden)
        expected[prefix + ".lora_up.weight"] = (output, rank)
        expected[prefix + ".alpha"] = ()
        expected.update({f"{prefix}.bridge.{key}": shape for key, shape in bridge.items()})
    return expected


def _bridge_shapes(stamp) -> dict[str, tuple[int, ...]]:
    # The shared bridge initializes random weights. A schema probe must leave
    # the caller's CPU RNG untouched, including when checkpoint validation fails.
    with torch.random.fork_rng(devices=[]):
        return {key: tuple(tensor.shape) for key, tensor in stamp.bridge().state_dict().items()}


def _expected_shapes(model, stamp) -> dict[str, tuple[int, ...]]:
    """Shapes from the live stamp bridge, not a copied MLP width table."""
    probe = _bridge_shapes(stamp)
    rank = int(stamp.spec["adapter_rank"])
    expected: dict[str, tuple[int, ...]] = {
        "particles": (int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]))
    }
    for name, module in attention_targets(model).items():
        prefix = "adapters." + name.replace(".", "-")
        expected[prefix + ".lora_down.weight"] = (rank, module.in_features)
        expected[prefix + ".lora_up.weight"] = (module.out_features, rank)
        expected[prefix + ".alpha"] = ()
        for key, shape in probe.items():
            expected[f"{prefix}.bridge.{key}"] = shape
    return expected
