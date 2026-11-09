# YuE2 PG0.9.5 numerical-response lifecycle

Root package0.4.0 pins core0.13.0 at `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, with published ParticleGAN0.9.5/tag1b8355b and release profile `release-0.9.5`. Public entrypoints remain `yue2_sliders.lifecycle:BACKEND` and `yue2_sliders.response_native:INFERENCE`; the shared core owns all game/update/optimizer/regularizer mathematics. The runtime delegates to `particle_sliders.pg095` and checks its complete source lock.

Default CPU fixtures use width32/auto. Genuine native2048-wide states require explicit continuous input domain, exact local assets and the strict native policy in [the native guide](docs/native-continuous-candidate.md). No projection, token generation or waveform rendering is introduced.

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
c = SessionConfig('yue2', 'fresh-yue2-pg095', dummy=True,
    rows=[dict(neutral='soft piano', positive='bright piano', lyrics='A new day'),
          dict(neutral='quiet flute', positive='strong flute', lyrics='A river runs')],
    steps=4, seed=7, batch=2, sample_seeds=1, history_tokens=4, ema_decay=None)
with open_response(c) as session:
    session.train()
    session.checkpoint('training.pt')
    session.export_inference('raw.pt', selection='raw')
```

New sessions disable EMA explicitly with `ema_decay=None` and use RAW weights
for training, evaluation, export and inference. Do not add EMA averaging,
comparison or selection. Historical EMA packets retain their original source
and configuration; they cannot resume into this fresh no-EMA configuration.

Use the same exact SessionConfig for continuation. The shared `particle-slider train --interface response` and `infer --mode response` APIs retain their interfaces. Native-only artifacts accept token IDs and exact device/revision/cache_dir runtime options; they return final AR hidden responses. They cannot train/resume, generate tokens or synthesize music.

The128×4 bank, rank8/alpha8 q/k/v/o geometry, frozen model and paired positive/neutral histories are unchanged. PGversion/profile/protocol/source identity is new. PG094 and source801 profiles refuse; old checkpoints/portable artifacts require their old source environment. [Historical lifecycle](docs/lifecycle-pg094.md) and [installation cohorts](INSTALL-COHORTS.md) retain those routes.

Omitted rates keep inherited optimized defaults. The current
[finished-audio decision](LEARNING-RATES.md) selects saved G0.00001 RAW64/h128,
using only the four-case equal-mean piano-stem attack-rise ON−OFF proxy. Reuse
that existing result, with its separator leakage limits; no extra confirmation
or replay is required. The earlier G0.0001 response-SSE screen and retained
native mechanical proofs have distinct scopes. This documentation changes no
source, checkpoint, defaults or response API.
