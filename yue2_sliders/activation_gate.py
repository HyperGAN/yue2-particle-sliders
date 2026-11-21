"""Calibrated YuE2 semantic gate using the shared steering equations."""

from dataclasses import asdict
from hashlib import sha256
from types import ModuleType
from zipfile import ZipFile

import torch


CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MODULE = "particle_sliders/adaptive_steering.py"
CORE_MODULE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"


def _shared_steering(core_wheel):
    # Keep the installed core identity used by the original RAW owner. This
    # standalone module has no intra-package imports and needs no global alias.
    with ZipFile(core_wheel) as archive:
        source = archive.read(CORE_MODULE)
    if sha256(source).hexdigest() != CORE_MODULE_SHA256:
        raise ValueError("YuE2 requires the exact shared steering module from core PR #269")
    module = ModuleType(f"{__name__}._core_{CORE_REVISION}")
    exec(compile(source, f"{core_wheel}/{CORE_MODULE}", "exec"), module.__dict__)
    return module


class SemanticGatePlan:
    """Explicit settings for the frozen YuE2 semantic feedback embedding."""

    def __init__(self, *, core_wheel, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(character not in "0123456789abcdef" for character in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the YuE2 gate calibration record")
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random gate does not use a YuE2 direction")
            self._direction = None
        else:
            if direction is None:
                raise ValueError("YuE2 CFP and CLABS require a fitted semantic embedding direction")
            probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (probe.ndim != 1 or not torch.isfinite(probe).all()
                    or not torch.isclose(probe.norm(), probe.new_tensor(1.0))):
                raise ValueError("YuE2 requires a finite unit direction in the native embedding space")
            self._direction = probe
        self._steering = _shared_steering(core_wheel)
        self._clabs = self._steering.CLABSParameters(**clabs) if clabs is not None else None
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._calibration_sha256 = calibration_sha256
        self._new_gate()

    def _new_gate(self):
        return self._steering.ActivationGate(self._method, self._parameter, seed=self._seed, clabs=self._clabs)

    def _projector(self, model, strength):
        if self._direction is None:
            return lambda _ids: None
        embedding = model.get_input_embeddings()
        if (not isinstance(embedding, torch.nn.Embedding)
                or embedding.embedding_dim != self._direction.numel()):
            raise ValueError("Direction width must match the resident YuE2 input embedding")
        direction = self._direction.to(device=embedding.weight.device)

        def project(ids):
            # YuE2Model consumes this embedding as the initial AR residual.
            # The adapter affects the subsequent attention projections.
            residual = embedding(ids)[0, 0]
            return float(self._steering.activation_projection(residual, direction, strength=strength))

        return project

    def _receipt(self):
        return {
            "core_revision": CORE_REVISION,
            "signal": "frozen YuE2 input embedding of the last sampled semantic token",
            "calibration_sha256": self._calibration_sha256,
            "method": self._method,
            "parameter": self._parameter,
            "seed": self._seed,
            "clabs": asdict(self._clabs) if self._clabs is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
        }
