# Fresh YuE2 integration audit — 2026-10-21

## Inspected sources and preservation

Current GitHub read-only inspection confirmed:

| Source | Head | Treatment |
|---|---|---|
| main | b33794d81b92c125891a011531f55554d65e2082 | inventory only |
| open draft [PR4](https://github.com/HyperGAN/yue2-particle-sliders/pull/4) | a9d09c7b74173360e459861eb5fe319c45d7c0c0 | selected integration base |
| open [PR3](https://github.com/HyperGAN/yue2-particle-sliders/pull/3) | cf150bb4533a60d15e33dfcb24fb4b2322ad35ed | retained untouched in origin ref; not merged |
| shared milestone | fd89e1c780ab5210cd1be98ec3b70977e6349379 | isolated read-only local clone, detached |

The owner's task-2/shared checkout was not edited. The exact lifecycle,
derivative handoff and HyperGAN/HNDL seam docs were read at this milestone.
The Iris reference factory and packaging were inspected. Shared core changes
remain owner/parent work.

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
  no duplicate GAN math. Registered through the current core entry-point group.
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

## Common interface rough spots for parent

1. **Modality contract:** lifecycle `API`, registry, backend errors, evaluate
   docstring and reference-image flags are image-named. Current capability data
   lacks explicit modality/response_kind fields. YuE2 sets audio/AR-hidden metadata,
   disables render and advertises `render_outputs=()`. Please add common explicit
   modality fields and a neutral registry/packet strategy, retaining existing
   image cohort identity. Do not call this hidden-response API a waveform API.
2. **Native PG profile gate:** `Backend.open()` unconditionally calls
   `_released_sources()` with the release default, before the factory. Native
   width2048 requires `alz-continuous-v1`/`continuous`, whose bytes necessarily
   differ. The lower-level `require_audio_width` and game already validate the
   exact explicit profile. Common preflight needs to honor the qualified pair
   without silently selecting/applying it. This draft preserves the guard and
   declares native lifecycle opening blocked pending that common fix.
3. **Checkpoint cleanup:** common `checkpoint_cleanup()` only restores two
   gradient-checkpoint attributes. Audio binding wraps `forward` and sets
   `_pg09_scale_checkpoint`, so the product factory snapshots those too. A common
   owned callback/handle from `checkpoint_with_adapter_scale` would remove this
   derivative-specific recovery code, including partial setup failures.
4. **Compact inference:** current session reconstructs prepared game/history and
   critic even for inference. Existing product native safetensor loader has its
   own strict geometry/provenance. Await owner compact-loading design before
   adding a native waveform callback; never hide caller assembly or skip identity.
5. **Run rendering/HNDL:** HyperGAN remains the run presentation owner via
   on_step/manifest/artifact seams. Current `AudioResponseCritic` is
   Linear(width+condition,16) → SiLU → Linear(16,1), raw logits. Product
   `RoutedProjection` owns low-rank projections and calls the shared routed bridge.
   These definitions need common HNDL matched-weight parity, parameter names,
   gradients and update/resume checks before replacement. No critic flattening,
   new network law or pretrained-backbone rewrite was introduced here.
6. **Error text:** render=false currently says native solver parity is required;
   that is not YuE2's waveform qualification reason. Allow backend-specific
   unavailable-render diagnostics in the common capability contract.

## Pending verification and limits

Initial authorization permits source inspection/light edits only. No test suite,
model/torch execution, dependency installation, build, GPU/model download, public
write, merge, deployment, permission/system change or active pin update ran.
Only stdlib AST/JSON/TOML parsing, wheel archive hash inspection and Git whitespace
checks were used. No product wheel/source-free execution receipt is fabricated.

Parent must schedule one bounded CPU slot after common interface selection:
new installed wheel outside source trees; exact 4 vs 2+2 resume with metrics,
EMA/RNG/coverage; invalid packet/optimizer/sampler/cohort/EMA restoration before
mutation; raw/EMA export-load; 0/.5/1 responses and negative/reference/render gates;
observation RNG/modes/buffers; staged setup/update failure cleanup; installed
origin/dependency guard and missing registration diagnostics. Run common
source-direction guard too. No native acoustic behavior is qualified by a tiny
fixture. Product legacy tests need their own original CDF environment; they must
not be stamped as passing under core0.7.

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
- Native profile acceptance and modality/compact-inference/HNDL contracts are
  **pending common-owner changes**, not claimed implemented by this derivative.
