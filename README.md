Current root package: **0.4.0 / core0.15.2 / published ParticleGAN0.11.1**, pinned to `ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` (Python>=3.11). See [installation cohorts](INSTALL-COHORTS.md) and [response lifecycle](LIFECYCLE.md). Native2048-wide responses require explicit `release-0.11.1/continuous`; default auto keeps its width gate. CPU conformance does not qualify new native training or transfer historical LR results.

The separate HNDL preview and historical controls below retain their original sources, environments and checkpoint identities.

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
`yue2-particle-sliders` is the YuE2 product repo in the Anima / Krea2 layout. It owns Hub ids,
the Comfy gate, YuE2 train and infer, prompt cards, and sample cards.

The [archived trainer and model controls](https://github.com/HyperGAN/yue2-particle-sliders/blob/108d127625c77bae952fc7c957bce3d6f5d48569/REPRODUCE.md)
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

`comfy_yue2.py` registers **YuE2 Particle Slider (ntc-ai)** under `NTC/YuE2`.
It checks a local slider against the winning formulation and does not render
audio inside the graph. See [COMFYUI.md](COMFYUI.md).

## Layout

| Path | Role |
|---|---|
| `yue2_sliders/` | AR backend, routed slider hooks, train, infer |
| `yue2_sliders.lifecycle` | Current training entry point for the shared CLI |
| `configs/yue2/` | Run configs, prompts, `model.lock.json`, sample cards |
| `comfy_yue2.py` | Comfy gate |
| `particle-sliders-core` | Shared engine, routed bridge and audio adapters |

Formulation exams stay in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
`V2_SPEC` is `analysis/slider2d/yue2_gmix_v2_exam.py` there.

Learning-rate controls and preserved defaults: [LEARNING-RATES.md](LEARNING-RATES.md).

## Opt-in cached fresh initialization

The separate [scoped owner API](scoped/README.md) adds explicit fresh, whole-FULL resume and RAW hidden-response inference in its own installed distribution. Four tiny installed-owner software cases passed with one update; this does not qualify new pretrained/acoustic execution. Original registrations, saved cohorts, historical EMA and learning-rate decisions remain unchanged. Authentic completed cache and owner admission are required.
