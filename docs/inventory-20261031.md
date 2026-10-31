# YuE2 inventory update — 2026-10-31

Public base: `0cdc1901a0636e12ee1898270156544101a53867`.
This branch reuses installer repair `b3c5daee1fc0b2fdf4f177510ca39fda0f27e171`.
The original source checkout and the unrelated retrain draft PR3 are unchanged.

Changes: current requirements delegate to the exact packaged core0.8.2 Git pin;
original requirements remain verbatim in `requirements-legacy.txt`. Wheel/sdist
assets now include the original configuration/catalog bytes without maintaining
a duplicate source copy. Installed catalog/sample-card helpers work without a
checkout. Streaming model-file hashing supports the declared Python3.10 API.
Native ownership accepts an explicit projection/factory so the HNDL cohort can
reuse the conflict-aware cleanup ledger.

The separate [HyperGAN/HNDL candidate](hypergan-hndl-cohort.md) provides actual
YuE2 network execution through the verified shared HyperGAN develop/HNDL builder,
with all GAN updates in the shared core. Its preview builder gives the new
cohort its own exact dependency metadata and Python range. It does not alter the
supported default pins, historical source locks, receipts or checkpoint labels.

## Validation

The task-local evidence directory is
`/ml2/codex-storage/future-workspaces/2026-10-31/task-12/evidence/`.
Tests are copied byte-for-byte outside the source checkout, then collected with
`python -I -m pytest --import-mode=importlib -c /dev/null` against installed wheels.

- Default core0.8.2/official PG0.9.1: 17 installed CPU cases, covering lifecycle,
  split resume, corrupt-checkpoint rejection without mutation, native-only raw/EMA
  reload, ownership conflicts/retry, captured gradients, frozen-base gradient
  ownership, exact zero-strength routing bypass, installed assets/entrypoints,
  Python3.10 digest API and refusal of the newer core in the old cohort.
- Preview core0.12/official PG0.9.4/exact HyperGAN develop/HNDL0.10: 8 installed
  CPU cases, covering matched-weight YuE2 responses/bank VJPs, critic scores and
  first/second derivatives, real updates, exact split resume, EMA export/reload,
  builder tamper rejection before attachment, deferred checkpoint backward after
  cleanup, native-only reload without training imports/construction, preview
  metadata and native2048/profile refusal before pretrained loading.
- An actual core0.8.2 training packet is refused by the new HNDL cohort without
  game/optimizer/EMA/RNG mutation. Both isolated dependency checks pass.
- Four existing documentation contracts are checked separately. The current
  wheel and sdist configuration bytes and historical requirements/lock identities
  are compared with the public-base Git objects: all 44 configuration/catalog
  files and the canonical lifecycle lock are unchanged. The default sdist builds
  a wheel with the same data; both package variants build.

Runtime: Python3.12.15, Torch2.14.1+cpu. Isolated virtual environments reuse a
read-only selection of existing CPU library files; they are not pristine native
environments. Core, ParticleGAN, HNDL and product wheels are installed separately;
HyperGAN uses the exact existing clean develop build with verified provenance.
No source-checkout path is inserted for the product/shared modules. Python3.10
itself was not executed; its missing file_digest API is simulated by the digest
regression. Initial harness failures were corrected before the final run.

## Qualification and release boundary

No pretrained/GPU/audio run, training campaign, model download or weight
publication occurred. Native2048 hidden responses still require the separately
owned explicit continuous-vector PG profile; published release0.9.4 does not
supply that admission. No gate is relaxed and no native response is reshaped.

The joint core4abc0eb is not public yet. Preview metadata records this exact
candidate and cannot resolve from public GitHub until coordinated publication.
A revised shared candidate needs a newly verified source lock and rerun tests.
Default source installation remains core0.8.2/PG0.9.1. Old training packets need
the original source environment; no checkpoint migration is provided.

The repository has no active CI/release workflow. This source PR changes no
weights, remote model artifacts or release tag and triggers no configured package
release or deployment. A merge is a source change, not native qualification or
permission to publish the preview package. Root owns review, Ada routing and merge.
