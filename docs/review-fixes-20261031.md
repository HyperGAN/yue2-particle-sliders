# Review corrections — 2026-10-31

Review target was PR6 head `69cda08ea5c61630b841fb69d97ef3199bfa0452`.
These fixes preserve the supported default and historical locks and keep shared
core4abc0eb, HyperGAN5f368a, HNDL0.10 and all width admission unchanged.

| Finding | Verified outcome and correction | Concrete evidence |
|---|---|---|
| Default package exposes the modern backend | Confirmed. Default pyproject now registers only `yue2`; its wheel omits `hndl_response.py`, the HNDL graph and runtime lock. Those sources live under `cohorts/hndl/yue2_sliders/`, and only the preview builder injects them. | Installed inventory and a fresh isolated process load every default registration while forbidding modern imports. No `yue2-hndl` entry point is discoverable. |
| Builder string replacements need count guards | Confirmed. All six substitutions go through `replace_once`, requiring exactly one occurrence before its sole `str.replace` call. | Twelve regressions remove/duplicate each individual anchor; a thirteenth verifies deterministic output and untouched input metadata. |
| Core wheel hash was recorded but unused | Confirmed. Runtime validation now checks the lock's wheel SHA256 against pip's installed archive provenance. Exact Git repository/commit provenance is the explicit source-install alternative. Neither skips the full actual-source closure or import-origin checks. | Wrong archive hash, missing archive hash, absent provenance and wrong Git commit refuse; exact Git provenance still requires the real qualified source bytes. |
| Wrong-source test changed cohort metadata | Confirmed test-coverage gap. Keep the separately named wrong-cohort case and add two actual-source cases. | One modifies real copied `pg09.py` bytes and observes refusal before hooks/flags/weights change. Another changes an actual copied HNDL graph file, opens the shipped factory with that source and verifies old-checkpoint refusal without game/optimizer/EMA/RNG mutation. No cohort field is edited in these source tests. |
| Test shipped `prepare()` and `ConditionalPG09` | Existing registry tests already exercised `prepare()` for the optimized preset, whereas the additional native API fixture used the explicit alias. Coverage is now explicit for both public presets. | `test_shipped_prepare_split_resume_ema_inference_and_wrong_cohort_rejection` goes through installed `open_response`, asserts the resulting exact common `ConditionalPG09` class and checks full/split resume plus raw/EMA native inference under both presets. The lower-level actual native-class fixtures remain supplemental. |
| `.to(float32)` might cast frozen BF16 base | Not reproduced. Shared HNDL projection keeps `module.forward` as a bound method; the host is not a registered child of the slider. No shared-core change is needed. | A frozen BF16 registered host retains parameter IDs, data pointers, BF16 dtype and byte-equal state through slider construction, explicit FP32 conversion and owner cleanup. Slider/base parameter IDs are disjoint; every adapter parameter is FP32; zero-strength BF16 response is exact. This is an internal CPU dtype/ownership test, not native BF16/GPU admission. |

Final review-fix test lanes:

- Default installed core0.8.2/PG0.9.1: **18 passed**.
- Modern installed core0.12/PG0.9.4/HyperGAN develop/HNDL0.10: **17 passed**, three
  native API cases deliberately excluded in this environment.
- Full preview suite in the pinned native-runtime dependency slot: **20 passed**.
- Source documentation and all builder guards: **17 passed** (4 + 13).
- Three dependency checks and wheel/sdist identity/package-content checks pass.

Task receipts and exact commands are under
`/ml2/codex-storage/future-workspaces/2026-10-31/task-12/evidence/review-fixes/`.
Installed tests are copied verbatim outside the checkout and use
`python -I -m pytest --import-mode=importlib -c /dev/null`.
The original head's evidence remains separate; these source changes require new
preview packets rather than relabeling the old preview cohort.

Joint core publication and the native2048 explicit continuous-vector profile
remain pending. No source/profile/width guard was bypassed; no pretrained/GPU,
waveform or quality qualification, merge, package release or deployment occurred.
