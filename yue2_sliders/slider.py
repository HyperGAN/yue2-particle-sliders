"""Routed AR adapters for YuE2.

Each projection owns a low-rank pair plus a bridge from ``stamp.bridge()``.
The bridge class is not reimplemented in this repo. Scale 0 skips the adapter
and returns the frozen projection exactly.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from yue2_sliders.backend import architecture_of, attention_targets


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
