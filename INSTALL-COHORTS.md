# Installation cohorts

The default package is now `yue2-particle-sliders` **0.3.0**, with Python>=3.11,
shared core **0.12.0** pinned to
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, and official ParticleGAN **0.9.4**.
`requirements.txt` delegates to this same immutable package requirement.
A fresh environment and new cohort are required; old source-bound checkpoints
are not rebound. The [current lifecycle guide](LIFECYCLE.md) documents the
`yue2` response training and native-only inference entry points.

```bash
python3.11 -m venv .venv-pg094
. .venv-pg094/bin/activate
python -m pip install .
particle-slider --help
```

`lifecycle-runtime.lock.json` verifies the current exact core archive or Git
provenance, all packaged source/data bytes and official PG release sources.
Native width2048 remains refused before loading the model. CPU numerical fixture
checks do not establish pretrained/GPU/acoustic qualification.

The optional [HNDL preview](docs/hypergan-hndl-cohort.md) builds package
`0.4.0.dev0`, pins the same core with its `hypergan-hndl` extra and installs only
`yue2-hndl` entry points. It requires Python>=3.11,<3.15. Keep it in its own
environment; do not install both product wheels together.

The historical CDF trainer requires its original core0.6.0/PG0.9.1 environment,
`requirements-legacy.txt`, original source checkout and immutable
`configs/yue2/shared-runtime.lock.json`. The previous core0.8.2/PG0.9.1 lifecycle
lock is archived at `configs/yue2/lifecycle-pg091.lock.json`, with its guide at
[docs/lifecycle-pg091.md](docs/lifecycle-pg091.md). Reproduce that lifecycle from
source commit `0cdc190` or the corresponding original reviewed source, not by
installing this current package into the old environment. Historical packet
sources, samples, publication metadata and model weights retain their identities.
