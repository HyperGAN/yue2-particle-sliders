# Replay the calibrated Crisp Piano CLABS gate

On four held-out Crisp Piano RAW256 cases, CLABS averaged +0.00747 over
OFF and +0.02447 over its equal-dose constant, which it beat on three
cases. All four pairs passed independent audio integrity checks. CFP
scored below its own constant on the same panel.
[Held-out evidence](crisp-piano-raw256-activation-gate-evidence.json)

Use the [replay settings](crisp-piano-raw256-clabs-replay.json) with the
recorded RAW256 checkpoint and its resident YuE2-3B AR owner. The preset
contains the fitted 2,048-dimensional direction, calibrated bias, feedback
statistics, native sampler settings, and four matched requests. The
calibration reached 198 of 398 eligible decisions on separate runs at the
declared 50% target. Build or supply a core wheel from
[core PR #269](https://github.com/HyperGAN/particle-sliders/pull/269) for
the YuE2 gate. The calibration receipt records the
[inclusive budget solver](https://github.com/HyperGAN/particle-sliders/commit/7b26e5b06f5db91e8d3f51a1602d4e459f0f6a22)
used to accept the measured firing count.

Run the example from the repository root after attaching the original RAW
adapter. Set `raw_path` to that checkpoint and `core_269_wheel` to the gate
wheel. Keep the original eager BF16 AR runtime and numerical profile.

```python
from hashlib import file_digest
import json
from pathlib import Path

import torch
from yue2.protocol import SongRequest
from yue2_sliders.activation_gate import SemanticGatePlan
from yue2_sliders.scheduled_semantic import generate_gated

preset = json.loads(Path("docs/crisp-piano-raw256-clabs-replay.json").read_text())
case = preset["cases"]["dev0-seed42"]
with Path(raw_path).open("rb") as raw_file:
    raw_sha256 = file_digest(raw_file, "sha256").hexdigest()
if raw_sha256 != preset["raw_sha256"] or pipe.weights != preset["pipeline_weights"]:
    raise ValueError("Crisp Piano replay requires the recorded RAW256 and YuE2 weights")

raw = torch.load(raw_path, map_location="cpu", weights_only=True)
resident = network.state_dict()
if (
    raw["weights"] != "raw"
    or raw["ema_decay"] is not None
    or raw["step"] != preset["raw_step"]
    or raw["network"].keys() != resident.keys()
    or any(tensor.dtype != resident[name].dtype
           or not torch.equal(tensor, resident[name].detach().cpu())
           for name, tensor in raw["network"].items())
):
    raise ValueError("The attached adapter must match the saved RAW256 tensors")

gate_plan = SemanticGatePlan(
    core_wheel=core_269_wheel,
    method=preset["method"],
    parameter=preset["parameter"],
    direction=preset["direction"],
    clabs=preset["clabs"],
    seed=int(case["gate_seed_decimal"]),
    calibration_sha256=preset["calibration_sha256"],
)
request = SongRequest(
    style=case["style"],
    lyrics=case["lyrics"],
    seed=case["seed"],
    cot=preset["native"]["cot"],
    cfg_scale=preset["native"]["guidance"],
)
strength = preset["nominal_strength"]
sampling = preset["native"]["semantic_sampling"]
native = pipe.effective_config(request, semantic_sampling=sampling)
if native["runtime_sha256"] != preset["native"]["runtime_sha256"]:
    raise ValueError("Use the recorded YuE2 native runtime")
device = torch.device(pipe.device)
if device.type != "cuda" or device.index is None:
    raise ValueError("Use the original CUDA AR owner with its explicit device")
with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
    torch.random.default_generator.manual_seed(case["seed"])
    torch.cuda.default_generators[device.index].manual_seed(case["seed"])
    with network.scaled(strength):
        plan = pipe.plan(request=request)
        semantic, trace = generate_gated(
            pipe, plan, scale_adapter=network.scaled, gate_plan=gate_plan,
            strength=strength, sampling=sampling,
        )
```

The gate seed is stored as a decimal string to preserve the full integer.
The initial semantic prediction is OFF. The next 199 eligible predictions
share each decision across both CFG branches. Save the returned trace and
use the same native NAR and VAE with adapter strength zero for the
finished-audio comparison.
