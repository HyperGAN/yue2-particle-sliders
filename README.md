> Fresh core0.8.2 / PG0.9.1 neutral response work is an untested, isolated [integration draft](LIFECYCLE.md).
> The CDF documentation below remains historical/legacy; no active environment is repinned.

# YuE2 Particle Sliders

Routed-particle composition sliders for [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B),
with the VAE at [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae).
`yue2-particle-sliders` is the YuE2 product repo in the Anima / Krea2 layout. It owns Hub ids,
the Comfy gate, YuE2 train and infer, prompt cards, and sample cards.

Training now calls `particle_sliders.ConditionalPG09` for the public ParticleGAN
**0.9.1** GAN-only recipe. `winning_formulation()` supplies the existing routed
adapter geometry, including `stamp.bridge()`; it does not select the training
objective. This repo does not vendor `RoutedMLP`, `GradRegularizer`, or ParticleGAN.
The legacy architecture recipe is `anneal-routed-particle-error-yue2-v1`.
See [FORMULATION.md](FORMULATION.md).

This draft uses the installed **0.6.0 candidate wheel** from commit
`cdfb80a76b882b4b73861b12f555263bd64ffeb7`. The wheel SHA256, all installed
shared Python/JSON hashes, and exact PG 0.9.1 wheel identity are locked in
`configs/yue2/shared-runtime.lock.json`. No shared checkout is needed at runtime.
This local candidate is not claimed as a public release. See
[REPRODUCE.md](REPRODUCE.md) and [the CDF follow-up](docs/pg091-cdf-adoption.md).

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

## Train and render

CPU checks need a fresh environment with the exact locked PG 0.9.1 and core wheels:

```bash
python scripts/train_yue2.py --dummy --steps 3 --until 1 --save_dir /tmp/yue2-first
python scripts/train_yue2.py --dummy --steps 3 \
  --resume /tmp/yue2-first/metal-yue2-particles_checkpoint.pt --save_dir /tmp/yue2-resumed
python scripts/infer_yue2.py --dummy \
  --weights /tmp/yue2-resumed/metal-yue2-particles_last.safetensors \
  --output_dir /tmp/yue2-cmp
```

AR q/k/v/o adapters and the actual routed bank train. NAR and VAE stay frozen.
Only +1 teachers are trained; qualified probe strengths are 0 through 1.
Negative inference remains unqualified. Native 2048-wide hidden states require
the separately reviewed explicit continuous-domain source profile; the released
PG default rejects them before pretrained model loading. Native GPU execution
and audio qualification remain with the coordinator and its admitted runner.

The eager runtime pin is `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`.
Details: [docs/yue2-slider.md](docs/yue2-slider.md) and [REPRODUCE.md](REPRODUCE.md).

## Comfy

`comfy_yue2.py` registers **YuE2 Particle Slider (ntc-ai)** under `NTC/YuE2`.
It checks a local slider against the winning formulation and does not render
audio inside the graph. See [COMFYUI.md](COMFYUI.md).

## Layout

| Path | Role |
|---|---|
| `yue2_sliders/` | AR backend, routed slider hooks, train, infer |
| `scripts/train_yue2.py`, `scripts/infer_yue2.py` | CLI entrypoints. Training calls `ConditionalPG09`; `winning_formulation()` supplies adapter geometry. |
| `configs/yue2/` | Run configs, prompts, `model.lock.json`, sample cards |
| `comfy_yue2.py` | Comfy gate |
| `particle-sliders-core` 0.6.0 candidate | Installed shared PG09 engine, routed bridge and audio adapters. |

Formulation exams stay in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
`V2_SPEC` is `analysis/slider2d/yue2_gmix_v2_exam.py` there.
