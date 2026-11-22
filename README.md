Current root package: **0.4.1 / core 0.15.2 / ParticleGAN 0.11.1**, pinned to `ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` (Python>=3.11). See [installation cohorts](INSTALL-COHORTS.md) and [response lifecycle](LIFECYCLE.md). Native2048-wide responses require explicit `release-0.11.1/continuous`; default auto keeps its width gate. CPU conformance does not qualify new native training or transfer historical LR results.

The original HNDL preview and published audio controls use the source revisions in the [reproduction guide](REPRODUCE.md).

Current finished-audio decision: use existing **cell1 G0.00001 RAW64** at fixed
D0.018 / bank0.030000000000000002, batch2/horizon128. Its equal-case frozen
estimated-piano attack-rise ON−OFF mean is **+0.014077682560999588** over four
matched cases, best of three saved rates. [Exact common options and scope](LEARNING-RATES.md).
This piano-stem proxy can contain separator leakage/artifacts; it is not human
listening or general acoustic-quality validation. Reuse the existing search,
RAW checkpoints, audio and scores; no new search, rerating or extra confirmation.
The older G0.0001 RAW128 teacher-SSE score0.5525904019655886 remains diagnostic
history. Defaults and historical release weights are unchanged. Historical
training EMA0.9 stays recorded; selection/inference use RAW, and new sessions
explicitly set `ema_decay=None` without averaging or an EMA comparison.

# YuE2 Particle Sliders

**Calculated memory:** Native BF16 AR response training: **14 GiB GPU / 24 GiB host** at batch 2/256 tokens. The conservative full-length eager music pipeline has a separate **92 GiB GPU** budget. [Full memory ledger and limits](MEMORY.md). These are planning estimates, not measured peaks.

Routed-particle composition sliders for [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B),
with the VAE at [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae).
`yue2-particle-sliders` owns YuE2 response conditioning, AR adapter wiring, Hub IDs, prompt cards, and sample cards.

The [archived trainer and published model controls](REPRODUCE.md)
require the original product revision and dependency environment.

Published weights and the listening gallery:
[ntc-ai/yue2-concept-sliders](https://huggingface.co/ntc-ai/yue2-concept-sliders)
· [live demo](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders).

## Samples

Each Off/On pair uses the same neutral caption, lyrics, and seed. Only strength
changes. These are short excerpts from the `particle-1200-v1` release (seed **1709**,
strength **1**). The index is `configs/yue2/sample-cards.yaml`.

| Control | Matched MP3 samples |
|---|---|
| Female voice | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/female/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/female/row0-seed1709-on.mp3) |
| Metal | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/metal/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/metal/row0-seed1709-on.mp3) |
| House | [Off](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/house/row0-seed1709-off.mp3) · [On (+1)](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/samples/particle-1200-v1/house/row0-seed1709-on.mp3) |

Sixteen voice and genre controls have train and eval sheets in `configs/yue2/catalog/`.
Original particle weights and ordinary distilled LoRAs are different files. The
links above are the original particles. Distillation notes stay on the
[Hub distillation guide](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/DISTILLATION.md).

## Comfy

For the **YuE2 Particle Slider (ntc-ai)** node, follow the [archived Comfy setup](COMFYUI.md).

## Layout

| Path | Role |
|---|---|
| `yue2_sliders/` | AR backend, routed slider hooks, and shared response APIs |
| `yue2_sliders.lifecycle` | Current training entry point for the shared CLI |
| `configs/yue2/` | Run configs, prompts, `model.lock.json`, sample cards |
| `particle-sliders-core` | Shared engine, routed bridge and audio adapters |

Formulation exams stay in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
`V2_SPEC` is `analysis/slider2d/yue2_gmix_v2_exam.py` there.

Learning-rate controls and preserved defaults: [LEARNING-RATES.md](LEARNING-RATES.md).

## Opt-in cached fresh initialization

The separate [scoped owner API](scoped/README.md) adds explicit fresh, whole-FULL resume and RAW hidden-response inference in its own installed distribution. Four tiny installed-owner software cases passed with one update; this does not qualify new pretrained/acoustic execution. Original registrations, saved cohorts, historical EMA and learning-rate decisions remain unchanged. Authentic completed cache and owner admission are required.

## Scheduled native semantic strength

With a resident eager pipeline and its attached RAW AR adapter, call
`yue2_sliders.scheduled_semantic.generate(pipe, plan,
scale_adapter=network.scaled, step_gains=gains)`. Use a CoT-off plan.
Guided requests apply the same gain to both native branches at each token.
Provide one gain per possible semantic token, including prefill.
Effective strengths must lie in `[0,1]`. The returned strengths
stop at native EOS or the token budget. Use the same native NAR and decoder
for finished-audio comparisons.

The archived Crisp Piano RAW256 profile gained +0.01585 mean intended
effect across four matched clips using late-heavy strength. Constant
scored −0.09635 and early-heavy scored −0.57201 at the same nominal
strength 0.5. The winning profile used `[0.5] * 100 + [1.5] * 100` gains
at native guidance 1.01. It improved three of four cases over constant
and scored above OFF on all four. All twelve clips passed separate
native stereo integrity checks.
[Per-case evidence](docs/crisp-piano-raw256-scheduled-strength.json)

### Opt-in activation gate

Pass an explicit `SemanticGatePlan` to `generate_gated` using an exclusively
owned resident eager pipeline and the same attached RAW AR adapter. Build or
supply the shared core wheel from the exact revision below. The gate loads
the byte-verified steering module from that wheel and the installed YuE2
response owner continues to use its original core pin.
[Shared core #269](https://github.com/HyperGAN/particle-sliders/pull/269)

```python
from yue2_sliders.activation_gate import SemanticGatePlan
from yue2_sliders.scheduled_semantic import generate_gated

gate = SemanticGatePlan(
    core_wheel=core_269_wheel,
    method="cfp",
    parameter=calibrated_threshold,
    direction=fitted_unit_direction,
    calibration_sha256=calibration_sha256,
)
semantic, trace = generate_gated(
    pipe, plan, scale_adapter=network.scaled, gate_plan=gate,
    strength=selected_nominal_strength,
)
```

The prompt prefill runs at strength zero, so its first prediction consumes no
gate decision. For each subsequent prediction, CFP and CLABS read the frozen
input embedding of the previously sampled semantic token before its attention
projections. Both CFG branches use the conditional branch's single decision at
that position and retain distinct native caches. Native EOS closes the trace.
The gate controls the saved RAW attention residual. The native acoustic stage
reads the semantic result after the adapter scope closes.

Fit the unit direction in this exact pre-attention embedding space on separate
target and neutral semantic trajectories. Bind the base revision, RAW SHA,
captions, lyrics, seeds and sampler to the calibration record. On a disjoint
split, run candidates with steering active and aggregate `fired` and `eligible`
from their traces. Calibrate CFP's threshold or CLABS's bias at the declared
budget using the shared core protocol. For CLABS, supply `method="clabs"`,
the calibrated bias, measured projection statistics and configured controller
settings as `clabs`, plus an explicit `seed`. The matched random control uses
`method="random"`, the declared firing rate and a seed. Save the trace with
the replay. It includes feedback tokens, projections, decisions, applied
strengths and realized strength sum.
[Core calibration and control protocol](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)

The measured Crisp Piano RAW256 CLABS profile beat its equal-dose constant
by +0.02447 averaged over four held-out cases. It passed independent audio
integrity on all four. Use the original checkpoint, sampler and recorded
gate settings for this opt-in profile.
[Calibrated replay](docs/crisp-piano-raw256-clabs-replay.md) ·
[Per-case evidence](docs/crisp-piano-raw256-activation-gate-evidence.json)
