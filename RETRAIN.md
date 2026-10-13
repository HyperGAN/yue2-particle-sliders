# Retrain v3: Metal and House AR sliders

**Status: prepared, NOT run.** This needs Mikkel's OK before it touches a GPU.

## Why
On the live Space (`particle-gmix-1600-v2`), AST never tags metal as Heavy metal (≤0.002 on both seeds, strengths 0 to 2). House lands on seed 1709 only (0.095). Keeping the slider on through the acoustic stage changed nothing, which means genre gets set by the AR-stage tokens. So the fix goes into the AR-side slider weights. This is step 3 of the genre-strength plan.

## What changed vs v2 (`configs/yue2/retrain-v3/`)
| | v2 (`particle-gmix-1600-v2`) | v3 |
|---|---|---|
| Rows | 4 catalog rows (`configs/yue2/catalog/{metal,house}-train.yaml`) | the same 4 rows plus 4 new ones |
| New rows' neutral | none | conflicting-genre captions: the v1 Space electronic default ("Genre: electronic song … synth chords … synth bass") and the current neutral Space default, each with 2 lyric sheets |
| New rows' positive | none | spells out timbre. Metal: distorted rhythm guitar, palm-muted riffs, double-kick drums, no synths. House: four-on-the-floor kick, offbeat open hats, 120–126 BPM, piano/organ stabs |
| Steps / LRs | 1600 · g 6e-4, d 9e-4, particle 6e-3 | the same |
| Seed bank | 4 × 128 = 512 | 8 × 64 = 512 (same bank size and step cost) |
| trained_scales | (1.0,) | **(1.0,), unchanged. [1.0, 1.5] is not supported.** In core `a119ca1`, `trained_scales` is a formulation key. A product config that restates it gets rejected (`_reject_forked_config`), and `train_step` hard-codes `network.scaled(1.0)`. Training at 1.5 needs a stamp change in HyperGAN/particle-sliders and then a pin bump. |

Note: the repo's `config-yue2-metal.yaml` is the older Arm-B 1200-step config. The v2 release ran the catalog prompt sheets for 1600 updates (`noise_decay_steps` = 1600, the weights are `*_step1600`), so v3 uses 1600.

## Launch (one GPU per genre, official runtime env per REPRODUCE.md)
```bash
cd yue2-particle-sliders && git checkout retrain/metal-house-ar-v3
. .venv/bin/activate   # env with requirements.txt + yue2-infer @ ef1936f
python scripts/train_yue2.py --config_file configs/yue2/retrain-v3/config-yue2-metal-v3.yaml \
  --save_dir models/metal-yue2-particles-v3 --device cuda:0
python scripts/train_yue2.py --config_file configs/yue2/retrain-v3/config-yue2-house-v3.yaml \
  --save_dir models/house-yue2-particles-v3 --device cuda:1
```
Add `--cache_dir <path>` if YuE2-3B is already local. Add `--allow_hub` only if a download is approved.

## Expected cost (estimates)
- GPU: an A6000 48 GB class card, like the pop-os pair the v2 campaign ran on. VRAM: *estimate* ~20–30 GB. The 3B AR runs in bf16, plus the adapter, critic and a 512-source bank. Inference uses a 20 GiB budget by default.
- Wall time: *estimate* **3–6 GPU-h per genre**, including gate renders. Measured basis: the v2 runs from 0 to 1600 took 1.8–5.7 h (median 2.5 h), and House took about 3.9 h. Both genres together: about 6–12 GPU-h, or about 3–6 h wall time on two cards.

## Ship gate (all must pass)
1. **AST**: render seeds **1709 and 2903** × **both captions** (the v1 electronic default `retrain-v3/neutral-electronic-v1-default.txt` and the Space neutral default `retrain-v3/neutral-space-default.txt`) at strengths 0 and 1. Score with `slider-week/genre-audit/analyze.py`. The target label (Heavy metal / House music) must **rise by ≥ 0.15, or become the top genre**, in all 4 cells.
2. **Human listen**: Mikkel listens to every cell. AST scored a metal clip near zero that Mikkel heard as metal, so AST alone can't ship or reject.
3. **Release reference**: the v3 release-sample measurements must stay at or above the v2 reference (`release_v2_measure.json`) on the target label and the health/clipping checks. No regression on the held-out eval prompts.

If v3 fails the gate, keep v2 live. Nothing goes to the Hub or the Space without a separate OK.
