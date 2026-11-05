# Installation cohorts

The root package0.4.0 requires Python>=3.11, core0.13.0 at `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, and published `particlegan==0.9.5`. `requirements.txt` installs this same package. Use a new environment and a fresh training cohort.

```bash
python3.11 -m venv .venv-pg095
. .venv-pg095/bin/activate
python -m pip install .
particle-slider --help
```

The source lock checks core archive/Git provenance, every packaged source/data byte, and the published PG0.9.5 source manifest. Native continuous vectors use explicit `pg_source_profile="release-0.9.5"`, `penalty_input_domain="continuous"`; see [native guide](docs/native-continuous-candidate.md). Auto remains width-gated.

PG094/PG801 packets require their original product/core/source environment, including measured source7757474/core5d68473. The old [lifecycle](docs/lifecycle-pg094.md) and [PG801 native guide](docs/native-pg801.md) remain historical. No packet is converted or rebound by this update.

The separate [HNDL preview](docs/hypergan-hndl-cohort.md), CDF/core0.6/PG091 trainer, archived PG091 lifecycle, published16 controls and audio samples retain their original pins. Do not install the new root package into those environments. NewCPU tests are software evidence, not pretrained native execution or an LR measurement on0.9.5.
