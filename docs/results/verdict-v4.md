# YuE2 retrain v4 — AST ship gate verdict

**Verdict: FAIL — metal FAIL, house FAIL.** Do not ship v4. No cell passes the gate (Δ ≥ 0.15 or target becomes top genre).

## Provenance
- Core: HyperGAN/particle-sliders PR #137 @ `84e0a0af3c7a2a1432e759fcd0a549497634a006` — https://github.com/HyperGAN/particle-sliders/pull/137
- Product: HyperGAN/yue2-particle-sliders PR #3 @ `d64b369bcf79e647f22e4ef641ed6541389ec430` — https://github.com/HyperGAN/yue2-particle-sliders/pull/3
- Stamp: `particle-gmix-1600-hold04-v1` (opt-in core stamp; noise_hold_ratio 0.4 instead of v3's 1.3). Gate renders loaded it explicitly via `locked_stamp(formulation_id=...)`.
- **Training batch: adv_batch 4 (v3 used 8).** Halved to keep GPU0 under 44 GB (the batch-8 trial peaked at 44.06 GB). Everything else matches v3: 1600 steps, same prompts, same LRs, seed 7.
- Training on pop-os GPU0: metal 11:47 AM–1:24 PM MT, house 1:24–2:10 PM MT, both exit 0.

## Gate setup (same as v3)
- 12 cells: 4 baselines at s0 plus 8 slider-on at s1.0. Captions are neutral-electronic-v1 and neutral-space, seeds 1709 and 2903, 750 semantic tokens (about 30 s), cot off.
- Each genre loaded in its own process (metal, then house), which avoids v3's "slider already attached" error.
- Renders ran on pop-os GPU1 (card peak 24.9 GB of 49.1 GB, shared with work's autojev llama-server). Nothing was preempted, and GPU0 was untouched.
- Tagger: MIT/ast-finetuned-audioset-10-10-0.4593 on CPU, sigmoid probabilities averaged over 10 s windows.
- The v4 baselines are byte-identical to v3's (same sha256), so the off column is directly comparable.

## v3 vs v4
| genre (target) | caption | seed | off | v3 on | v3 Δ | v4 on | v4 Δ | v3 top on | v4 top on | v4 cell |
|---|---|---|---|---|---|---|---|---|---|---|
| metal (Heavy metal) | electronic | 1709 | 0.0002 | 0.0374 | +0.037 | 0.0050 | +0.005 | Progressive rock | Blues | fail |
| metal | electronic | 2903 | 0.0001 | 0.0031 | +0.003 | 0.0130 | +0.013 | Independent music | Electronic music | fail |
| metal | space | 1709 | 0.0004 | 0.0242 | +0.024 | 0.0531 | +0.053 | Progressive rock | Progressive rock | fail |
| metal | space | 2903 | 0.0001 | 0.0014 | +0.001 | 0.1157 | **+0.116** | Country | Progressive rock | fail |
| house (House music) | electronic | 1709 | 0.0021 | 0.0011 | −0.001 | 0.0021 | −0.000 | Electronic music | Soul music | fail |
| house | electronic | 2903 | 0.0057 | 0.0012 | −0.005 | 0.0007 | −0.005 | Electronic music | Soul music | fail |
| house | space | 1709 | 0.0005 | 0.0009 | +0.000 | 0.0014 | +0.001 | Electronic music | Soul music | fail |
| house | space | 2903 | 0.0001 | 0.0012 | +0.001 | 0.0018 | +0.002 | Electronic music | Soul music | fail |

Top genre with the slider off is Lullaby or New-age music in every cell.

## Read
- **Metal moved up but still fails.** The mean Δ went from +0.016 (v3) to +0.047 (v4). The best cell is now +0.116 (space, 2903), which is about 3× v3's best (+0.037) but still under the 0.15 bar. Three of four cells now tag Progressive rock or Blues, so the slider pushes toward rock, but Heavy metal never becomes the top genre. The electronic/1709 cell went down from v3 (+0.037 to +0.005).
- **House shows no effect.** Every Δ is between −0.005 and +0.002, the same as v3 and still worse than v2's 0.095 on seed 1709. The on clips shift from "Electronic music" (v3) to "Soul music" (v4), never toward House.
- Reminder: on 2026-10-08 Mikkel heard a metal clip and said it sounded metal, and AST is known to under-score metal. A listen to `metal_v4_space_s1.0_seed2903.flac` (the strongest AST cell) is worth doing before reading too much into these scores.

## Training notes
- Metal v4: losses sat at about 0.693–0.70 for most of the run, with an early wobble (d range 0.21–1.28, g max 4.48) and 168/1600 logged rows more than 0.02 off 0.693 (v3 had 101). Steps 1500–1600 averaged d 0.693 / g 0.693.
- House v4: same flat pattern, 182 rows off 0.693, ending at d 0.694 / g 0.693. Unlike v3 house, which ended in a discriminator blow-out (steps 1500–1600 averaged d 0.001 / g 7.97), v4 house stayed stable.
- The `vic` term again collapsed early, to about 1e-5 by mid-run (v3 was similar).

## sha256
Checkpoints:
```
5c764b185d4ee34d185d50cd217a20526622197797179aca3ddc5b282b3418b4  metal-yue2-particles-v4_1600.safetensors
fa7f9a92b0aceb9469ee3e9a65ef0b803c6c47def01f2d7f4c59f6ba48da0e52  house-yue2-particles-v4_1600.safetensors
```
Clips:
```
d4d6f14f9406ac9621a3a7a6b60d40e449de06764c94f3efd29d0cc559b5931e  baseline_electronic_s0_seed1709.flac
01ee490bc39d7caa053443fbe705adcde4f6b4761bbcb7669845f3e93dfd76b6  baseline_electronic_s0_seed2903.flac
12385ebc43b7e1c8c44be37c8ef3af3c3d807123bebd230cbe0c20a9358fd033  baseline_space_s0_seed1709.flac
35cf472f22db7072ce61cfe25748bfcf0fcdfb9fa03dac82c2047828d371bfc4  baseline_space_s0_seed2903.flac
9688072e8856272a1e815fb0cc40ccbd3e8c2017d26ed9aac406a7ea222484bc  house_v4_electronic_s1.0_seed1709.flac
1cc2978d0d171a62bfd838184df8b027a3f259ac6083191585426eb8d461f30d  house_v4_electronic_s1.0_seed2903.flac
74307840f210f9cfe5d4849f4e6c68f7e0226fa240e27ea70249cc603582a984  house_v4_space_s1.0_seed1709.flac
7a8e1796bf55076f92be0acb0f014d78eff69f21eff3866f2ee6a0d7e2ce584a  house_v4_space_s1.0_seed2903.flac
2be6c01e68476b3d71ede9c0baea6f7efb976330deaaf66684bc67c9fa51d659  metal_v4_electronic_s1.0_seed1709.flac
a3d206834788a9c59cade30ade484c19674afe4dbff2d942262bb4990e96a43f  metal_v4_electronic_s1.0_seed2903.flac
a4a6227e6d0b69005d56f2a71de1470d78067673955a848c9ffffff894b97564  metal_v4_space_s1.0_seed1709.flac
41c5c529cfa9961ffc111706dae422374035a98782205f482dd913e6ff3888e0  metal_v4_space_s1.0_seed2903.flac
```

No merge, no Hub, no Space, no HF uploads were done.
