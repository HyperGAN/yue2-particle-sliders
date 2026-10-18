# YuE2 retrain v3 — AST ship gate verdict

**Verdict: FAIL — metal FAIL, house FAIL.** Do not ship v3.

- Weights: PR #3 `retrain/metal-house-ar-v3` @ e2c08e8, `*-yue2-particles-v3_1600.safetensors`
- Renders: pop-os GPU1, 12 cells (4 shared baselines at s0 + 8 slider-on at s1.0), captions neutral-electronic-v1 / neutral-space, seeds 1709 / 2903, 750 semantic tokens (~30 s), cot off
- Tagger: MIT/ast-finetuned-audioset-10-10-0.4593 on CPU, sigmoid probs averaged over 10 s windows
- Gate (RETRAIN.md): target score rises ≥ 0.15, or target becomes top genre, on both seeds

| genre | caption | seed | off | on | Δ | top genre on (off) | cell |
|---|---|---|---|---|---|---|---|
| metal (Heavy metal) | electronic | 1709 | 0.0002 | 0.0374 | +0.037 | Progressive rock (Lullaby) | fail |
| metal | electronic | 2903 | 0.0001 | 0.0031 | +0.003 | Independent music (New-age) | fail |
| metal | space | 1709 | 0.0004 | 0.0242 | +0.024 | Progressive rock (Lullaby) | fail |
| metal | space | 2903 | 0.0001 | 0.0014 | +0.001 | Country (Lullaby) | fail |
| house (House music) | electronic | 1709 | 0.0021 | 0.0011 | −0.001 | Electronic music (Lullaby) | fail |
| house | electronic | 2903 | 0.0057 | 0.0012 | −0.005 | Electronic music (New-age) | fail |
| house | space | 1709 | 0.0005 | 0.0009 | +0.000 | Electronic music (Lullaby) | fail |
| house | space | 2903 | 0.0001 | 0.0012 | +0.001 | Electronic music (Lullaby) | fail |

**vs v2:** v2 metal Δ ≈ 0.002; v2 house Δ 0.095 (seed 1709), 0.002 (seed 2903).
- Metal v3: best Δ 0.037 (electronic, 1709). That's up to ~18× v2, but still about 4× short of the 0.15 bar. On 1709 the top genre moves toward rock ("Progressive rock"), but it never becomes Heavy metal.
- House v3: Δ between −0.005 and +0.001. That's **worse than v2's 0.095 on 1709**. The slider turns the clips into generic "Electronic music" and doesn't push toward House.

Training note: the `vic` term collapsed to ~3e-9 by step 50 of metal, which may be relevant to the weak/no effect.

Run notes: run_gate_render.py was patched to accept .flac (YuE2 writes audio.flac), and it was relaunched once for the house cells after a "slider already attached" error. There was no GPU contention: GPU1 peaked at 27.3/49.1 GB, and Tau's alz-cal-chain-v43-20261014 stayed active with no errors.
