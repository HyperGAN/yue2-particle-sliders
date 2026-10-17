# YuE2 derivative PG 0.9 audit

Historical receipt for source-based head `9390a4b4000f4dd4cd3cd165aab93b4de4d2772a`.
Current installed-wheel work is documented in [PG 0.9.1 wheel adoption](pg091-wheel-adoption.md).

Repository identity: `https://github.com/HyperGAN/yue2-particle-sliders.git`.
Audit base: clean `main` at `b33794d81b92c125891a011531f55554d65e2082`.
Work is isolated on `audit/shared-pg09-20261017`. No active owner checkout,
installed environment, native source pin, admission card, checkpoint, service,
credential or permission was changed.

| Entry point | Result |
|---|---|
| `scripts/train_yue2.py`, `yue2.train.main/train` | Native-first CLI bootstrap; exact source dependency probe; sole shared ConditionalPG09 update engine; full resume |
| `scripts/infer_yue2.py`, `yue2.infer.main/infer` | Same source probe; strict pre-hook base/provenance/tensor validation; active-weight reload; deterministic dummy base |
| root Comfy `__init__.py`, `comfy_yue2.YuE2ParticleSlider.load` | Native-first custom-node bootstrap and dependency probe before stamp; same full tensor validation |
| direct `ParticleSlider.load` | Actual native base identity required; failed checks install no hooks |

## Demonstrated bugs fixed

The old trainer imported shared primitives but implemented its own legacy paired
RpGAN/VIC, optimizer and cyclic sampler loop. It never executed ConditionalPG09.
It wrote only EMA inference weights and had no critic/optimizer/RNG resume.
The new thin CLI uses the shared public PG 0.9 recipe, actual routed bank,
conditions, sampler, update/optimizer and transaction/checkpoint implementations.
It rejects inactive legacy learning-rate knobs and records the resolved recipe.

Native `yue2` was shadowed by the product package. Both CLI scripts and Comfy now
import installed native first and append only product modules, with collision
checks. Dummy use without a native install remains supported.

Trainer exports omitted actual base-model identity, so the old inference check
was bypassed. The new export binds real base identity and shared source/game
signature; loader validates it before installing hooks. Comfy previously checked
metadata without tensors; it now rejects malformed/missing/nonfinite tensor
payloads. Inference files are active adapter/bank state, full training checkpoints
are separate, and public shared exports cannot resume training.

`--dummy --steps` was lost when invoked through a real CLI with `argv=None`.
Explicit CLI steps now survive. Dummy inference formerly constructed a different
random base, and computed off deltas using whichever scale happened to appear
first. It reconstructs the original training seed and computes off independently
of scale order. Missing config paths are rejected instead of silently ignored.

Only +1 teachers train, with neutral students and shared sampled histories.
Conditions contain the complete frozen neutral response, prompt lengths, history
length and sign. Scales outside [0,1] are rejected; native render protocol is
CFG1, cot off, AR only, with NAR/VAE at zero. Negative training/effects remain
unqualified. Native continuous responses retain width 2048; release/auto rejects
that width before pretrained loading. The exact reviewed explicit continuous
profile is required for a separately admitted native run.

## Frozen dependencies and evidence

Core source: `3e94b13eb5d6cd1ceee3f018f019a68603f35696` at the explicitly supplied
`audio-core-pin-86fc4e0-20261017` checkout (the directory name is stale).
Internal audio source: `9262a930cd351459e93ac33cc542fa379c12fe52` at
`/mnt/nvme/sliders/audio-adapters-pg09-20261017`. Actual source hashes and import
origins are in [cpu-imports.json](../evidence/pg09-derivative/cpu-imports.json).
There is no copied backend implementation or guessed public Git dependency.

CPU interpreter `/mnt/nvme/sliders/qualification/envs/pg090/bin/python`,
Torch 2.11.0+cu128, installed PG 0.9.0: **60 passed in 24.22s**.
[Full JUnit](../evidence/pg09-derivative/full-cpu.xml),
[training cases](../evidence/pg09-derivative/training-cpu.xml),
[inference cases](../evidence/pg09-derivative/inference-cpu.xml), and
[commands](../evidence/pg09-derivative/commands.sh). Training/inference XMLs are
filtered views of the single exact full-suite run. Original product baseline:
19 passed in 4.80s, [receipt](../evidence/pg09-derivative/baseline-cpu.xml).

Regressions prove both actual public presets and their active resolved recipes,
GAN-only loss decomposition with MSE forbidden, complete trainable connectivity
and updates, real 128x4 bank single ownership, frozen base, exact scale 0, probe
RNG preservation, and active product weights. Three fresh CLI processes compare
uninterrupted 3 updates against1+2 continuation: full game state, optimizer,
metrics, probes and active safetensors match exactly. Changed conditions reject
before output creation. Clean Comfy subprocess has no core PYTHONPATH and proves
exact environment-selected source imports. Source/domain/protocol/provenance
failures reject before model or hook mutation as applicable.

Actual native environment `/ml2/sliders/.venv-yue2-retrain/bin/python`,
Torch 2.10.0+cu128: both CLI help commands and native model-class imports passed;
[native origins](../evidence/pg09-derivative/native-imports.json). Upstream runtime
commit is `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`. CUDA remained uninitialized.
No pretrained native model was constructed or rendered by this audit.

## Integration and remaining gates

Integration target is this repository's `main` via a draft PR. Parent routes
exact heads through Ada and verifies merges; this task does not merge.

The internal audio APIs are not included in the public core wheel. Task14 is the
sole extraction owner, targeting `particle_sliders.pg09_audio` and
`particle_sliders.pg09_backends`. The draft remains explicitly source-dependent
until its exact package head/wheel is independently reviewed. Moving class paths
or source hashes requires a new cohort or explicit reviewed migration; this head
is not a drop-in resume target for old native smoke checkpoints.

The previous owner smoke remains passed on its original frozen sources:
177.615s/exit0, actual native 3 vs 1+2, base unchanged, bank changed, exact reload.
That is owner evidence, not a native test of this derivative. Its Y20
continuation, accepted export hashes, admission/guardian cards, native render,
listening/acoustic parity, and bounded longer work remain pending with the audio
owner/coordinator. GPU admission belongs solely to coordinator
`01a147d7-f540-778e-ad79-4d8eb267403d`; no GPU work or model publication occurred.
