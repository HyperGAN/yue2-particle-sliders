> Historical PR6/a794 record. The active default and optional preview now pin
> coreddb4142/PG0.9.4; see [the current HNDL guide](hypergan-hndl-cohort.md).
> The receipts below apply only to their original a794 source.

# Modern preview migration to core a794

Only the active YuE2 HyperGAN/HNDL preview moves to the exact public shared-core
commit `a794e3b4a8813b012f2c4faa3da82b78d4a90214`. The product preview is now
`0.3.0.dev1`. Default `0.2.0` metadata, its core 0.8.2 / PG 0.9.1 lock, historical
requirements, model identities and recorded artifacts remain unchanged.

The regenerated modern lock binds all 51 recursively packaged source/data files,
without a suffix allowlist, and the reviewed core wheel SHA256
`22e47ced4af8cb874d818db2f4006738f5f978cfcff324693931b6ab23f21c8a`.
The lock generator rejects a different archive; regeneration is deterministic.
The version remains core 0.12.0 with exact official PG 0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0. Actual imported core
sources and Git provenance are verified in a new isolated public-source install.

The previous preview at product head `468e5a1e4311fbf19d33aff9b0c99598c7a79ae8`
continues to identify its original core 4abc cohort. Real training and raw/EMA
inference packets captured in that original environment are rejected by dev1.
Their bytes are unchanged. Rejection leaves new training state and inference RNG
unchanged; inference rejects before any native constructor is reached. Use the
original source/environment for those old packets; no compatibility relabeling
or automatic checkpoint migration is provided. Earlier review notes remain intact.

Validation uses installed artifacts with source checkouts absent from import paths:

- Fresh public-source wheel environment, Torch 2.6.0+cpu: 21 cases passed; three
  native API cases were deliberately excluded.
- Supplemental Torch 2.14.1+cpu library view: 21 cases passed; the same three
  native API cases were excluded.
- Supplemental pinned native API dependency view, Torch 2.10.0+cu128 executing
  on CPU: 24 cases passed. These use random tiny YuE2 classes, not pretrained weights.
- Fresh preview sdist installation: eight selected lifecycle, exact resume,
  raw/EMA reload, recursive tamper and cleanup/failure cases passed.
- Documentation and guarded preview metadata transformations: 17 cases passed.
- Fresh wheel/sdist dependency checks and supplemental Python pip checks pass.

The added migration checks compare exact OFF/raw/EMA responses after reload,
reject altered Python, HNDL and nested non-code data, retain foreign projection
ownership during failed close, retry cleanup successfully, and restore a staged
host plus RNG after partial setup failure. Existing bank architecture, objectives,
protocols, source checks and the native-width refusal are retained.

The preview builder produces both a wheel and an installable source distribution.
See [the current cohort guide](hypergan-hndl-cohort.md) for exact installation and
lock-generation instructions. Task-local literal commands, package identities,
provenance, old-packet refusals and logs are under
`/ml2/codex-storage/future-workspaces/2026-10-31/task-12/evidence/a794-migration/`.

The parent reported bounded synthetic GPU qualification of the shared core.
That is software-fixture evidence, not YuE2 native-width, pretrained, acoustic or
quality admission. This product migration ran CPU checks only. Native width 2048
remains refused until the separately reviewed continuous-vector profile is
admitted. No GPU job, umbrella change, HNDL optimization, model download, weight
publication, release or deployment occurred. The final source delta returns to
root for Claude actual-diff review before any landing decision.
