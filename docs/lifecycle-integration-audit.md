# Fresh YuE2 integration audit — 2026-10-21

## Inspected sources and preservation

Current GitHub read-only inspection confirmed:

| Source | Head | Treatment |
|---|---|---|
| main | b33794d81b92c125891a011531f55554d65e2082 | inventory only |
| open draft [PR4](https://github.com/HyperGAN/yue2-particle-sliders/pull/4) | a9d09c7b74173360e459861eb5fe319c45d7c0c0 | selected integration base |
| open [PR3](https://github.com/HyperGAN/yue2-particle-sliders/pull/3) | cf150bb4533a60d15e33dfcb24fb4b2322ad35ed | retained untouched in origin ref; not merged |
| initial shared milestone | fd89e1c780ab5210cd1be98ec3b70977e6349379 | retained historical proposal evidence |
| selected reviewed response line | 21761ca8fd059439b58aa56a69bf8d1647f50e31 | core0.8.2 / existing PG0.9.1; isolated clone now detached here |

The owner's task-2/shared checkout was not edited. The exact lifecycle,
derivative handoff and HyperGAN/HNDL seam docs were read at this milestone.
The Iris reference factory and packaging were inspected. Shared core changes
remain owner/parent work. The parent subsequently supplied independent installed
review of the response contract. Its exact canonical 091 lock was copied unchanged;
wheel, all 38 runtime-source hashes, and companion archive were verified read-only.
Three guides in the verified companion match the checkout; the immutable sdist's
missing-guide limitation is preserved rather than repaired under its old pin.

PR4 is the current actual GAN path, with original CDF/core0.6 and released
PG0.9.1 source checks. Its retained receipts qualify six clock and ten consumer
cases. Earlier 104/32-case receipts are explicitly older 4be evidence, not CDF or
this draft qualification. Existing native `b337`, audio `9262`, shared `3e94`
and passed-smoke lineage is retained. Y20, combined native CDF memory/render,
acoustic/listening and longer gates remain pending, as PR4 states.

PR3's result receipts say v3 and v4 failed their gate/do not ship; v3b4 evaluation
remains pending. Its configs, weight/clip hashes and hold results are not copied,
relabelled or replaced here. Its package rename commit `3fc7092` informed the
same mechanical namespace fix atop PR4; the rest of its payload is preserved in
its existing branch and left out of this new cohort.

## Entrypoint and ownership audit

- Existing train: `yue2_sliders.train.train` checks the old exact lock, loads native
  config before tensors, freezes native model, prepares shared-history rows,
  constructs the real routed bank and shared `ConditionalPG09`. Legacy full
  game checkpoints, active safetensors and adapter exports keep their formats.
- Existing resume: frozen-base digest and full source-bound shared game state;
  legacy entrypoint only. Existing published formats remain validated by their
  original loader. The namespace/source changes themselves create a new source
  cohort; old PR4 resume requires the original PR4 source checkout.
- Existing infer: safetensor provenance before hooks, actual native base identity,
  deterministic tiny comparisons, and a separate native YuE2 pipeline callback.
  CFG1/cot-off/AR-only placement is preserved; no sample/render work was run.
- Existing Comfy: validates the same product record/tensor payload; no waveform
  output is promised. It retains the old CDF dependency gate.
- New factory: full loader/slider/adapter/row bank/conditions/critic/game ownership,
  exact new milestone lock, shared session stepping/resume/export/evaluate/EMA,
  no duplicate GAN math. Registered through the separate neutral response entry-point group. A second native-only inference registration owns base+adapter construction without training imports or teacher rows.
  Installed qualification remains pending.
- Cleanup: snapshots native module forwards, checkpoint flags, modes and base
  requires_grad before hook installation. Setup failures, partial projection
  construction, session failures and close restore this state. No global patch
  or foreign site-packages injection is used.

Generator is `routed_particle_gan`: one real 128x4 bank owned once by the slider
and aliased into projection bridges, rank8/alpha8 q/k/v/o LoRA branches, shared
RoutedMLP bridge law. The existing shared response wrapper exposes the actual
trainables to PG; no fake particle tensor or adapter-GAN substitution is made.
Frozen base is separate, trainables FP32. Response is the genuine final AR hidden
state (tiny32/native2048), neutral student, frozen positive target, identical
neutral continuation history, full frozen neutral hidden conditioning plus prompt
lengths/history/sign. There is no diffusion sigma/scheduler, no categorical
fallback, projection, truncation or spatial reshape.

## Current common seams and remaining gaps

The initial source draft `045a7e72da52243160b59434984ce17d30a5b441` reported
image-named descriptors/packets, release-only session preflight and missing compact
inference. The parent-reviewed response line resolves those API issues. This
proposal now uses `ResponseBackend`/`ResponseCapabilities`, separate response
packets and registry, exact profile/shape admission, explicit refusal reasons and
native-only v2 inference. No image geometry or waveform semantics are imposed.

Remaining concrete boundaries:

1. **Native width:** official PG0.9.1 release/auto refuses native flat width2048;
   official PG0.9.3 has the same unsupported selector boundary. The old
   `alz-continuous-v1` source is unrecovered, so this new product proposal explicitly
   rejects selecting it. This is an upstream/profile recovery/parity gate, not a
   remaining neutral-contract defect. Pretrained constructors stay behind it.
2. **Checkpoint hook ownership:** common `checkpoint_cleanup()` restores two
   gradient-checkpoint attributes but does not return an undo handle for
   `checkpoint_with_adapter_scale`, which wraps forward and sets
   `_pg09_scale_checkpoint`. The owned product factory still composes a forward/
   flag snapshot with provider close. A common idempotent undo handle, including
   partial setup, would remove this derivative-specific recovery code.
3. **Observation ownership:** common scope now covers provider len/fetch and
   adapter state/RNG. Product observation additionally restores native buffers,
   registrations, modes and forward/checkpoint/gradient flags. Native hidden
   evaluation uses no KV cache; v2 evaluates supplied native integer prefixes,
   not a token sampler. No whole frozen-parameter snapshot or waveform claim.
4. **HNDL/presentation:** HyperGAN remains the on_step/manifest/artifact consumer.
   `AudioResponseCritic` retains Linear(width+condition,16) → SiLU → Linear(16,1),
   raw logits; routed projections retain rank8/alpha8 and the actual bank. These
   definitions still need separately coordinated matched-weight HNDL parity,
   parameter naming, gradients, updates and resume review before replacement.

The restored minimal shared audio closure is consumed through the canonical
package. New product code imports `pg09_audio`/`pg09_backends` and the neutral
lifecycle/inference seams; no campaign worker, legacy audio CLI or render stack
is used as a fallback. The provider reuses the existing product's token/history
preparation helper only during training construction. Native-only inference
imports none of the product train/provider/lifecycle modules.

## Pending verification and limits

Initial authorization permits source inspection/light edits only. No test suite,
model/torch execution, dependency installation, build, GPU/model download, public
write, merge, deployment, permission/system change or active pin update ran.
Only stdlib AST/JSON/TOML parsing, wheel archive hash inspection and Git whitespace
checks were used. No product wheel/source-free execution receipt is fabricated.

Parent must schedule one bounded CPU slot on the selected response line:
new installed wheel outside source trees; exact 4 vs 2+2 resume with metrics,
EMA/RNG/coverage; invalid packet/optimizer/sampler/cohort/EMA restoration before
mutation; raw/EMA export-load; 0/.5/1 responses and negative/reference/render gates;
observation RNG/modes/buffers; staged setup/update failure cleanup; installed
origin/dependency guard and missing registration diagnostics. Run common
source-direction guard too. No native acoustic behavior is qualified by a tiny
fixture. Product legacy tests need their own original CDF environment; they must
not be stamped as passing under core0.8.2. See [the explicit pending plan](response-session-test-plan.md).

## Action status

- Initial unprivileged GitHub reads failed DNS in the network sandbox. Approved
  read-only `gh pr list`, `gh pr view` and isolated `git clone` subsequently
  succeeded. No automatic-review rejection occurred. One `gh pr view` request
  used unsupported `baseRefOid`; it was corrected and the supported read succeeded.
- Sending the early plan to parent thread
  `01a157c0-24ba-7273-9be8-c36fa69e301f` via the app returned `thread not found`.
  This is a failed delivery, not an uncertain write. No resend/bypass attempted;
  this handoff contains the report for automatic parent notification.
- CPU build/install/test slots are **pending parent scheduling**, not denied.
- Native-width/profile recovery and HNDL parity remain **pending parent/upstream
  coordination**. Neutral response and compact-inference APIs are adopted in
  source; product runtime qualification remains pending.
- Read-only local fetch of the exact response commit printed a shallow-root-ref
  warning; checkout and exact source verification succeeded at `21761ca`. No
  owner checkout was edited, and no retry/escalation or public write occurred.
