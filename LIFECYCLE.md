# YuE2 current numerical-response lifecycle

The default `yue2-particle-sliders` package **0.3.0** uses shared core **0.12.0**
at `5d68473a098801ee1ab3ebbb12463daef51c6346` and default official **ParticleGAN 0.9.4**
at `330eb94ee748cdcad131b0275d8de75fd5a7f96f`. Python **3.11 or newer** is required.
Install with `python -m pip install .` or `python -m pip install -r requirements.txt`.
The core Git requirement selects this exact source commit; its metadata pins
`particlegan==0.9.4`. Use a new environment and a new checkpoint cohort.

The current entry points are `yue2_sliders.lifecycle:BACKEND` for response training
and `yue2_sliders.response_native:INFERENCE` for native-only numerical inference.
`particle-slider` comes from the pinned shared core. The game uses its public
`ConditionalPG09` compatibility name, which now executes the official PG0.9.4
`release-0.9.4/auto` objective, optimizers, regularizers and checkpoint lifecycle.
An opt-in native source candidate uses the exact upstream PG commit
`801ae1ead8f88b88eca8f5398137e3e1bc01ad36`, selected only by
`pg_source_profile="source-801ae1e-continuous-v1"` together with
`penalty_input_domain="continuous"`. It is distinct from the official same-version
wheel; a version-only replacement is refused.
No ParticleGAN objective or fallback formula is copied into this product.
The existing 128x4 routed bank, rank8/alpha8 q/k/v/o adapter geometry, frozen YuE2
backbone, positive/neutral histories and raw-logit audio critic are preserved.

The CPU fixture has response shape `[1,32]`. Native `m-a-p/YuE2-3B` produces
`[1,2048]`: official PG0.9.4 still rejects this continuous-vector width before
loading pretrained weights. We do not truncate, reshape or change its domain to
bypass that gate. The explicit source candidate admits the native width after
complete shared/upstream source and local model/tokenizer checks; see
[native source candidate](docs/native-continuous-candidate.md). Native GPU
execution remains unqualified until a separate measured proof. Waveform rendering, acoustic quality, pretrained/GPU training,
negative strengths and token generation remain unqualified. The separate
[HNDL preview](docs/hypergan-hndl-cohort.md) has the same native-width boundary.

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

config = SessionConfig('yue2', 'new-yue2-pg094-cohort', dummy=True,
    rows=[dict(neutral='a quiet acoustic song', positive='a loud guitar song',
               lyrics='[verse]\nA train rolls through the night'),
          dict(neutral='a soft piano song', positive='a bright brass song',
               lyrics='[verse]\nWe meet beside the river')],
    steps=4, seed=7, batch=2, sample_seeds=1, history_tokens=4, ema_decay=.9)
with open_response(config) as run:
    run.train(stop_after=2)
    run.checkpoint('new-run/training.pt')
with open_response(config, resume='new-run/training.pt') as run:
    run.train()
    run.export_inference('new-run/adapter-v2.pt', selection='ema')
    hidden_response = run.evaluate(index=0, seed=11, strength=.5)
```

Native-only inference builds the frozen base, adapter and shared AR response
without constructing teacher data, critic, game or optimizer. An exported
inference artifact cannot train or resume. Its source identity, base digest,
precision, architecture, context, PG profile and selected raw/EMA state are bound.

```python
from particle_sliders.inference import open_inference

with open_inference('new-run/adapter-v2.pt', options={'device': 'cpu'},
                    operation='response') as native:
    hidden_response = native.response(token_ids=[12, 21, 30], strength=.5, seed=11)
```

The current core CLI uses `particle-slider train --interface response` and
`particle-slider infer --mode response`. The native request is JSON
`{"token_ids":[12,21,30]}`: an existing, nonempty, in-vocabulary AR prefix within
its context limit. It evaluates the final hidden state; it does not synthesize
a song. Native runtime options are `device`, `revision` and `cache_dir`.

`lifecycle-runtime.lock.json` binds every packaged shared source/data file,
including nested package data and HNDL graphs, the exact core version and Git
commit, both explicitly selected PG wheel/source identities and source digests.
Core provenance must be the locked wheel archive or the exact Git commit;
missing provenance and altered bytes are rejected before native construction.
Regenerate deterministically from the reviewed artifact with
`python release_tools/lock_runtime.py /path/to/particle_sliders_core-0.12.0-py3-none-any.whl`.

The prior PG0.9.1/core0.8.2 lifecycle lock is retained byte-for-byte at
`configs/yue2/lifecycle-pg091.lock.json`, and its original guide is archived as
[PG091 lifecycle](docs/lifecycle-pg091.md). The CDF `configs/yue2/shared-runtime.lock.json`,
legacy CLI/Comfy training recipes, old packets, 16 published controls and their
sample assets keep their original source identities. Use the original source
checkout and environment to reproduce or resume them; no package bump retrains
or migrates old weights. See [installation cohorts](INSTALL-COHORTS.md).
