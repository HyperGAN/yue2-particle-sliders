# PG 0.9.1 installed wheel adoption

Qualified replacement candidate: `4be0b5b29d2e2c741431f468d71ba8a7c884b724`,
wheel SHA256 `2e7621c370e790f65702acf76cfa2f428b393cd3f863245aa26d8a76ff096ada`.
This is a proposed package candidate, pending Ada review and parent merge.

The product now imports `ConditionalPG09` and the public
`particle_sliders.pg09_audio`/`particle_sliders.pg09_backends` APIs from an
installed core wheel. Runtime no longer requires `conceptmod`, a shared
checkout, `YUE2_*_SOURCE`, or checkout PYTHONPATH. It validates exact wheel
identity, installed Python/JSON bytes, and actual import origins against
`configs/yue2/shared-runtime.lock.json` before use.

This is a new source lineage. Historical PG 0.9.0 native sources, admitted smoke,
checkpoints and resident environments remain untouched. Old full state is
rejected before mutation or continuation outputs; there is no automatic
migration. Full checkpoints and shared `.pt` exports also bind the selected
installation path. Moving them to another environment path requires explicit
qualification. Product safetensors retain actual base/stamp/package provenance.

Every training update still calls the single shared engine. Both public recipes
use the actual routed bank once, frozen positive teachers, neutral student
contexts, shared histories and explicit conditions. The host binds FP32 policy;
PG 0.9.1's upstream CUDA BF16/channels-last options are not qualified here.
Active safetensors, shared inference exports and full training state remain
distinct. Only +1 is trained, inference accepts [0,1], and no acoustic claim is
made from response tensors.

An additional product bug was demonstrated during adoption: both model-backed
and Comfy tensor-schema validation constructed a random bridge and consumed CPU
RNG, including rejected files. Eight cases failed before the fix. One shared
RNG-isolated bridge-shape helper now serves both validators; failed native loads
preserve weights, gradient flags and hooks as well as RNG.

## Provisional first-candidate evidence

Candidate `42a2e371c537180bd189fd4a288b640f9f1c98fc` wheel SHA
`7418efe0d89782b2fd266c6a8e536539243d06dd42ff5d393e409bf4b0f2c872` was installed
with official PG 0.9.1 wheel SHA
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec` in a new
task-local environment. All RECORD entries, package bytes and archive source
mapping were independently verified.

The initial [source-free suite](../evidence/pg091-wheel/candidate42-full.xml)
passed 74 cases in 23.58s. That is **provisional evidence, not final acceptance**:
a focused [optimizer restore regression](../evidence/pg091-wheel/candidate42-optimizer-rejections.xml)
subsequently failed four original-`bispf` cases: G/D negative second moment and
G/D missing second moment were accepted. Two optimized-role rejection cases
passed. The package owner has superseded this candidate and owns the shared
repair; the product does not duplicate that logic.

Synthetic 2048-wide product AR response tests run both public presets through
the separately staged exact 49-source PG 0.9.1 continuous profile. They preserve
native vector shape and prove positive continuous penalties, bank updates,
frozen base, exact resume and export reload. They use random CPU stand-ins,
not native pretrained weights or listening audio. Release/auto still rejects
width2048 before pretrained tensor loading.

## Final replacement evidence

A fresh `/ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-final`
installed official PG 0.9.1 and the exact replacement wheel from local artifacts.
No existing environment was modified. Runtime imports resolve the wheel's
`site-packages`; there are no shared checkout paths or legacy audio imports.
The installed 32 core Python/JSON files, 49 released PG sources, wheel RECORD,
67 archived source mappings and Git tree were independently verified.

**104 tests passed in 41.78s**, with zero skips/errors/failures and CUDA
uninitialized. [Full JUnit](../evidence/pg091-wheel/final-full.xml),
[training/package cases](../evidence/pg091-wheel/final-training-cpu.xml),
[inference cases](../evidence/pg091-wheel/final-inference-cpu.xml),
[commands](../evidence/pg091-wheel/commands.sh), and
[actual runtime/import/source receipt](../evidence/pg091-wheel/final-runtime.json).
Filtered XMLs are views of that single full-suite run.

The separate [32-case wheel/restore gate](../evidence/pg091-wheel/final-wheel-gates.xml)
passed in 15.68s. Negative, missing, deleted, emptied and extra started G/D
optimizer histories reject atomically. Valid initial-empty and honestly unused
critic histories resume exactly. Product tests preserve full receiver state,
checkpoint input bytes, and absence of continuation output on rejection. Both
presets continue correctly through the actual shared audio adapter call path,
and fresh-process 3-vs-1+2 compares full state, metrics, probes and active weights.
Full checkpoints, shared exports and product safetensors remain distinct.

The preserved native environment still passes both CLI help commands and native
model-class imports: [receipt](../evidence/pg091-wheel/native-imports.json).
That check creates no model and does not install the candidate in the native
environment. Combined native PG 0.9.1 execution, memory, pretrained rendering,
acoustic/listening parity, Y20 continuation and longer tests remain separate
coordinator/audio-owner gates. Existing passed PG 0.9.0 smoke lineage remains
unchanged and cannot silently resume into the new package lineage.

Integration target is the existing YuE2 draft PR #4 into `main`. Publication
uses an explicit full refspec, with read-only push-configuration inspection and
remote branch/main verification before and after. No merge, GPU job, active
source retargeting, credential/permission/service change or model publication
occurred. Parent routes exact tested heads through Ada before merge.
