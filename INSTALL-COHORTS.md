# Installation cohorts

The current packaged lifecycle uses `particle-sliders-core` 0.8.2 at
`f6807a91546e43e198645a3f8925d2dc7c9e03b4` with exact official ParticleGAN `0.9.1`.
Install from this repository root with `python -m pip install -r requirements.txt`.
The requirements entry delegates to `pyproject.toml`, so the two install paths
resolve the same immutable shared core. No unpublished PyPI core release is assumed.

`requirements-legacy.txt` preserves the previous installer verbatim for historical
trainer reproduction in a separate environment. Existing source locks, model
identities, checkpoints and receipts retain their original meaning. Do not combine
legacy requirements and the packaged lifecycle in one environment, or repoint a
historical lock to newer bytes. The product runtime still verifies its own
canonical source map; a successful installation alone does not qualify execution.

This repairs installation consistency. It does not adopt core0.11.0/PG0.9.3 or
PG0.9.4, migrate checkpoints, qualify pretrained/GPU output, or enable refused
continuous-response/render operations. Those require distinct reviewed cohorts.

Keep this PG0.9.1 environment separate from core0.11.0/PG0.9.3 and
HyperGAN develop, which currently requires PG0.9.4/HNDL0.10.
These Git dependency wheels are for source installation, not a PyPI release.

Do not inject a foreign site-packages tree or the historical dependency into
a live native environment. The packaged Python>=3.10 declaration is preserved;
this task exercised Python3.12 CPU fixtures only.
