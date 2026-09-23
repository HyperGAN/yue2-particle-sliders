# YuE2 Particle Sliders

Routed-particle composition sliders for [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B),
with the VAE at [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae).
`yue2-particle-sliders` is the YuE2 product repo in the Anima / Krea2 layout. It owns Hub ids,
the Comfy gate, YuE2 train and infer, prompt cards, and sample cards.

The shared game is `winning_formulation()` from `particle-sliders-core`, pinned at
[`4340e28bed388d50800c469525b460a108091da0`](https://github.com/HyperGAN/particle-sliders/commit/4340e28bed388d50800c469525b460a108091da0).
This repo does not vendor `RoutedMLP`, `GradRegularizer`, `locked_shared`, or ParticleGAN.
The provisional Hub recipe name is `anneal-routed-particle-error-yue2-v1`.
See [FORMULATION.md](FORMULATION.md).

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

CPU smoke (no Hub weights):

```bash
python -m pip install -r requirements.txt
python scripts/train_yue2.py --dummy --steps 2 --save_dir /tmp/yue2-smoke
python scripts/infer_yue2.py --dummy \
  --weights /tmp/yue2-smoke/metal-yue2-particles_last.safetensors \
  --output_dir /tmp/yue2-cmp
```

Live metal run, local weights or `--allow_hub`:

```bash
python scripts/train_yue2.py \
  --config_file configs/yue2/config-yue2-metal.yaml \
  --save_dir models/metal-yue2-particles
python scripts/infer_yue2.py \
  --weights models/metal-yue2-particles/metal-yue2-particles_last.safetensors \
  --style 'English, piano-led pop, clear close lead vocal, steady bass and dry drums.' \
  --lyrics_file lyrics.txt --scales 0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-example --allow_hub
```

AR q/k/v/o adapters train. NAR and the VAE stay frozen. Scale 0 is the base
model. The eager runtime pin is `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`.
Details: [docs/yue2-slider.md](docs/yue2-slider.md) and [REPRODUCE.md](REPRODUCE.md).

## Comfy

`comfy_yue2.py` registers **YuE2 Particle Slider (ntc-ai)** under `NTC/YuE2`.
It checks a local slider against the winning formulation and does not render
audio inside the graph. See [COMFYUI.md](COMFYUI.md).

## Layout

| Path | Role |
|---|---|
| `yue2/` | AR backend, routed slider hooks, train, infer |
| `scripts/train_yue2.py`, `scripts/infer_yue2.py` | CLI entrypoints. Both call `winning_formulation()`. |
| `configs/yue2/` | Run configs, prompts, `model.lock.json`, sample cards |
| `comfy_yue2.py` | Comfy gate |
| `particle-sliders-core` @ `4340e28` | Shared stamp. Not copied into this repo. |

Formulation exams stay in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).
`V2_SPEC` is `analysis/slider2d/yue2_gmix_v2_exam.py` there.
