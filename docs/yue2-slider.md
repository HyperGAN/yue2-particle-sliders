# YuE2 sliders

This product trains and renders composition sliders on `m-a-p/YuE2-3B`.
The shared game is `winning_formulation()` from `particle-sliders-core`
at `a119ca1ecd3d5d6c437065839d22739b04f2f4d8`. Read [FORMULATION.md](../FORMULATION.md)
before changing a knob.

Only AR attention trains: `model.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}`
(112 projections on the released 28-layer model). NAR, both MLPs, embeddings,
norms, the output head, and the VAE stay frozen. Scale 0 bypasses the adapter
and is the base model exactly. The trained range is `[0, 1]`.

The positive caption is the teacher (`lm_target: faithful_plus_neu` on the
stamp). The student reads the neutral caption. Prompts describe instruments,
voice, room, and timing. Named-reference syntax is rejected.

## Install

CPU checks:

```bash
python -m pip install -r requirements.txt
python scripts/train_yue2.py --dummy --steps 2 --save_dir /tmp/yue2-smoke
```

Live runs use a separate environment. The runtime pin is upstream commit
`ef1936f2ee39fe8de486a0f47a481c95f8d4da87`. See [REPRODUCE.md](../REPRODUCE.md).
Weights are CC BY-NC 4.0. This repo does not ship them.

## Train

```bash
python scripts/train_yue2.py \
  --config_file configs/yue2/config-yue2-metal.yaml \
  --save_dir models/metal-yue2-particles
```

Configs under `configs/yue2/` set the prompt sheet, step budget, and optional
model-surface fields (batch, history length, learning rates). Rank, particle
count, critic width, noise horizon, and auxiliary weights come from the stamp.
Putting them in YAML raises.

Exports are `.safetensors` plus a JSON sidecar. The tensor file is the EMA
shadow at `stamp.spec["ema"]`. Dummy exports are marked and cannot be rendered
as songs.

## Render

```bash
python scripts/infer_yue2.py \
  --weights models/metal-yue2-particles/metal-yue2-particles_last.safetensors \
  --style 'English, piano-led pop, clear close lead vocal, steady bass and dry drums.' \
  --lyrics_file lyrics.txt --scales 0,0.5,1 --seed 7 \
  --output_dir eval/listen/yue2-example --allow_hub
```

Every scale uses the same neutral style, lyrics, and seed. The adapter is on
for AR planning and semantic tokens, and off for NAR and the VAE. Use a fresh
output directory. `--dummy` writes `comparison.json` and does not write audio.

## Samples

Featured Off/On pairs from `particle-1200-v1` (seed 1709, strength 1) are
indexed in `configs/yue2/sample-cards.yaml` and linked from the README.
The sixteen control prompt sheets are `configs/yue2/catalog/*-train.yaml`.

## Where the exams went

2D gates and formulation crown notes stay in
[HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders/tree/a119ca1ecd3d5d6c437065839d22739b04f2f4d8):

- `analysis/slider2d/yue2_gmix_v2_exam.py` (`V2_SPEC`)
- `docs/yue2-gmix-v2-2d-scoreboard.md`
- `docs/yue2-particle-bridge.md`
- `docs/yue2-gan-stability.md`, `docs/yue2-gan-toy-audit.md`, `docs/yue2-c9-native-trial.md`

Those files are not copied here.
