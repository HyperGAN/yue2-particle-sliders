# Parent-scheduled YuE2 CPU qualification plan

Status: pending resource admission; **none of these commands/tests has run**.
Exact shared line: core0.8.2 `f6807a91546e43e198645a3f8925d2dc7c9e03b4`, PG0.9.1
release/auto. Use the canonical owner wheel/lock and companion source archive.
Build only the proposed product head, then use a new isolated installed target
outside both product/shared source trees. Existing environments remain untouched.
No source PYTHONPATH, model loading/downloads or GPU execution is required.

Parent controls memory/time/thread limits and the serial slot. Proposed flags:
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
plus one Torch interop thread set in the isolated driver before imports execute.
No full legacy suite belongs in this slot: legacy tests require their old CDF
line. The first product cases are now authored in `test_response_ownership.py` and
`test_response_sessions.py`; they remain unrun. Their 13 parametrized cases should
be split into two serial, offline phases, each with a proposed 90-second wall bound
and 2GiB process/group cap, one compute/BLAS/Torch interop thread, subject to parent
admission. Fresh-process inference workers each have a 45-second inner timeout.
Run only these two files initially against the newly built installed product wheel.
This document is a plan, not a passing-test stamp.

1. Verify product/core/PG installed origins, exact versions, canonical 38-file
   source map and complete released PG sources; CUDA stays uninitialized. Run
   shared dependency-direction guard against installed product metadata/imports.
2. Resolve response and inference entry points. Prove no image-registry fallback,
   response-v1 vs image-v1 packet refusal before prepare, and helpful missing/
   duplicate registration diagnostics.
3. With two synthetic sound-only rows and real routed 128x4 bank, execute four
   updates vs fresh-process two+two continuation under a fixed four-update recipe.
   Compare full game/optimizer/coverage/RNG, metrics, EMA, exported bank and adapter
   values exactly. Verify frozen base unchanged and each participating trainable
   owns one parameter, including actual bank alias ownership.
4. Reject changed cohort/config/base/checksum/sampler/optimizer/source/EMA before
   mutation; retain usable live session after rejected restoration. Refuse old
   CDF/published product packets through the response loader.
5. Verify exact strength0 bypass against the same frozen base, .5 and 1 responses,
   unsupported negative/>1/nonfinite strengths, reference and render gates before
   allocation. Require response metadata/audio AR geometry and no waveform claim.
6. Verify eager provider identity and detached row/condition copies. Repeated
   seeded observation, invalid bounds, provider fetch failure and native forward
   failure preserve ambient/training RNG, modes, persistent/nonpersistent buffers,
   native buffer registrations and adapter state. Training coverage is unchanged.
7. Force failures after partial projection installation, checkpoint wrapping,
   provider capture, critic/game creation and a failed update. Confirm all owned
   forwards/flags/modes/gradient ownership return to their original state and
   provider closes. Repeated successful close is safe. Add native ownership nesting/conflict/retry:
   failed wrapper handles remain retained; successfully closed siblings are not
   replayed; broader projection/native cleanup waits; resolve the external owner
   and retry the binding callback after the common session has marked itself closed. Borrowed staged-device/shape mismatch
   must fail before RNG/device allocation.
8. Export native-only raw and EMA v2. In a fresh process, remove access to training
   JSON/rows and forbid product lifecycle/provider/train imports plus critic/game/
   optimizer construction. Load only the native registration and base+adapter;
   compare same-prefix raw/EMA responses to selected training values. Confirm
   origin/selection/schema receipts and refusal to train/resume from v2.
9. Test new-prefix native integer token IDs: empty/bool/negative/out-of-vocabulary/
   over-context requests refuse without forward. `device/revision/cache_dir` are
   the only runtime options; reject injected rows/steps/game settings. Validate
   changed native architecture/seed/precision/identity before adapter mutation.
10. CLI smoke in fresh output directories: capabilities, split train/resume,
    explicit v1/v2 files and hashes, numerical evaluation and native-only response
    output. Never invoke a waveform renderer.
11. Advertise native2048 then forbid all pretrained constructors: release/auto
    must reject before allocation. Explicit unrecovered continuous profile must
    fail closed. Do not run a patched profile, alter release bytes, select another
    PG line, truncate hidden width or change the objective.

Proposed source-free CLI commands (`$PY` is the admitted installed interpreter;
`config.json` is the two-row dummy SessionConfig from LIFECYCLE.md):

```bash
$PY -I -m particle_sliders.session_cli capabilities --interface response --config config.json
$PY -I -m particle_sliders.session_cli train --interface response --config config.json --out first --stop-after 2 --export-format both --selection ema
$PY -I -m particle_sliders.session_cli train --interface response --config config.json --resume first/training.pt --out resumed --export-format both --selection ema
$PY -I -m particle_sliders.session_cli evaluate --interface response --config config.json --artifact resumed/adapter.pt --strength .5 --out hidden.pt
$PY -I -m particle_sliders.session_cli infer --artifact resumed/adapter-v2.pt --runtime runtime.json --mode response --native prefix.json --strength .5 --out native-hidden.pt
```

`runtime.json` is `{"device":"cpu"}`; `prefix.json` is
`{"token_ids":[12,21,30]}`. Record exact product head, wheel/source hashes,
commands, exit status, counts, resource peaks, installed origins and CUDA state.
No native/tokenizer/acoustic qualification follows from these tiny CPU checks.


Proposed first pytest selections (admitted installed target, copied tests outside
source trees, importlib mode and plugin autoload disabled):

```bash
$PY -I -m pytest -c /dev/null --import-mode=importlib -o pythonpath= -q test_response_ownership.py
$PY -I -m pytest -c /dev/null --import-mode=importlib -o pythonpath= -q test_response_sessions.py
```

A parent driver must set both Torch thread counts before pytest collection and
supply exact metadata-owned installed targets to child processes. No collection,
model construction or execution has occurred during the source-only MOSS pilot
window. Rerun remaining planned contract cases only after these meaningful first
checks and when additional failures/changes justify them.
