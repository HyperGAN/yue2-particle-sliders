# Retrain v3 / v4 gate results (do not ship)

Both retrains failed the RETRAIN.md audible gate (target AST rise >= 0.15 or top genre on both seeds). Ember review: v4 do-not-ship.

| run | commit | stamp (noise hold) | batch | metal best delta | house delta | verdict |
|---|---|---|---|---|---|---|
| v2 (shipped) | model repo 31649319 | particle-gmix-1600-v2 (1.3) | 8 | ~0.002 | 0.095 (seed 1709) | current |
| v3 | e2c08e8 | particle-gmix-1600-v2 (1.3) | 8 | +0.037 | -0.005..+0.001 | FAIL |
| v4 | d64b369 | particle-gmix-1600-hold04-v1 (0.4), core PR #137 @ 84e0a0a | 4 | +0.116 | -0.005..+0.002 | FAIL |
| v3b4 (matched baseline) | d64b369 | particle-gmix-1600-v2 (1.3) | 4 | not gated | not gated | trained 2026-10-14; eval pending Mikkel |

Noise hold per run is confirmed from step-1600 log sigma: v3b4 1.47 / 1.55 (metal / house), v4 0.45 / 0.48.

Diagnosis: the vic term reaching ~0 is not a particle collapse (std per dim stays ~1.04, which already satisfies the target). D/G losses stay at chance (~0.693) for the whole run in v3, v4 and v3b4, so the critic gives the slider almost no signal. Lowering the noise hold alone did not fix that.

Per-run details, checkpoint and clip sha256s: `verdict-v3.md`, `verdict-v4.md`, `scores-v*.json`. Receipts pack on pop-os: `/ml2/sliders/receipts/yue2-v4-v3b4-20261014T1949.tar.gz` (sha256 953bdc8beeef0a9fa6295834f25ee52e09b34c420705c9563ea13230c2956979).
