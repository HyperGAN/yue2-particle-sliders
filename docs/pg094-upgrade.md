# Official PG0.9.4 package upgrade — 2026-11-04

The canonical default package is now `yue2-particle-sliders` 0.3.0 with
Python>=3.11, shared `particle-sliders-core` 0.12.0 at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, and official ParticleGAN0.9.4
(`330eb94ee748cdcad131b0275d8de75fd5a7f96f`, `release-0.9.4/auto`).
The core pin includes merged unified training/execution PR161 as well as the
current install/PG094 fixes from shared-core PR162.

Both active locks bind all **574** packaged core source/data files recursively,
without a suffix allowlist. The reviewed core wheel SHA256 is
`0a78d4f88e0c5e5436acfb9f3aca2ae75a7889d9b4f6525cda4a38c42e39257f`.
Exact archive or Git commit provenance is mandatory; both installed test slots
used the exact Git source pin. Lock generators reproduce the same source maps.
The separately built HNDL preview is now 0.4.0.dev0 with the same current core,
HyperGAN develop5f368a3/HNDL0.10 and its separate `yue2-hndl` registrations.

Completed bounded CPU checks, with CUDA hidden, Hub offline and OMP/BLAS=1:

- **30 passed** outside the checkout against the installed default product wheel:
  official PG094 updates for both presets, frozen base, exact split/resume,
  raw/EMA native-only reload, projection/checkpoint ownership and safe close
  retries, source/data tamper rejection, archive/Git provenance, installed assets
  and backend discovery. The real unified execution API used an isolated
  installed response worker and wrote periodic checkpoints and final artifacts.
- **21 passed, 3 skipped** outside the checkout against the installed HNDL preview:
  real shared HNDL builder/critic parity, gradients, official PG094 game,
  resume/export/inference and source/provenance/ownership checks. The three tiny
  native-architecture fixtures need the separate pinned `yue2-infer` source slot;
  it was not downloaded or substituted.
- **74 passed, 77 skipped** in the source suite. The historical CDF runtime cases
  are explicitly marked for their immutable core0.6/PG091 environment; the
  installed-only, optional HNDL and unrecovered continuous-profile cases also
  retain their declared environment boundaries.
- The normal wheel and separate HNDL wheel/sdist built successfully. The archived
  PG091 lifecycle lock is byte-for-byte unchanged (SHA256
  `12e2310b2680173b5a6c3c220270d3d79cb9029d5660bf23f7d6e6f06fd986f3`).
  Historical CDF locks, model/catalog/sample assets and 16 published controls
  were not changed or relabeled.

These are software/conformance checks with a width32 fixture. Official PG094
continues to refuse native continuous-vector width2048 before pretrained loading.
No model downloads, GPU training, wide-domain bypass, waveform rendering,
pretrained/acoustic quality qualification, weight retraining or publication is
claimed. Old CDF/PG091/a794 packets retain their original source and cohort;
reproduction/resume uses that original source checkout and environment.
