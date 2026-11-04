# YuE2 optional HyperGAN/HNDL preview

This separately built `yue2-particle-sliders` **0.4.0.dev0** cohort pins current
shared core0.12.0 at `ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, official PG0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0 through
the shared `hypergan-hndl` extra. It requires Python>=3.11,<3.15.

It preserves the existing real shared HNDL projection builder and native hidden
critic from PR6; no new dummy particle prior or local GAN implementation is
introduced. It exposes only the `yue2-hndl` response and native-only inference
registrations. The default `yue2` package remains a separate routed-bank cohort.
Both now use the same official PG0.9.4 game and current exact shared-core pin.

```bash
python release_tools/build_hndl_preview.py /absolute/output/directory
# In a separate fresh environment:
python -m pip install /absolute/output/directory/yue2_particle_sliders-0.4.0.dev0-py3-none-any.whl
```

The shipped lock binds all current core source and nested data files and accepts
only its reviewed wheel or exact Git provenance. Regenerate it with
`python release_tools/lock_hndl_runtime.py /path/to/particle_sliders_core-0.12.0-py3-none-any.whl`.
Each new run needs a new cohort. Packets from a794 or any older source retain their
original source identity and are not automatically migrated.

Native width2048 is still rejected by official PG's continuous-vector gate before
pretrained loading. Neither matching tiny architecture fixtures nor CPU numerical
checks qualify pretrained/GPU/acoustic results. This upgrade does not retrain the
16 published controls. The immutable PR6 source and older verification record are
preserved in [hypergan-hndl-a794.md](hypergan-hndl-a794.md) and
[a794-modern-migration.md](a794-modern-migration.md); those receipts describe their
original source, not this new package.
