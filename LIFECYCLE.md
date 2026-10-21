# YuE2 lifecycle integration draft

This isolated draft is based on open PR4 at
`a9d09c7b74173360e459861eb5fe319c45d7c0c0`. It proposes an owned factory against
shared milestone `fd89e1c780ab5210cd1be98ec3b70977e6349379` (core 0.7.0).
It has **not been executed, built, installed or qualified**. Existing environment
pins, `requirements.txt`'s CDF pin, checkpoints and receipts are unchanged.
See [the audit and pending gates](docs/lifecycle-integration-audit.md).

The new installable `yue2_sliders.lifecycle:BACKEND` constructs the entire game.
It preserves the existing native `YuE2Backend`, frozen neutral/positive teacher
contexts with shared continuation history, real routed particle bank, projection
hooks, `YuE2ARResponseAdapter`, `AudioResponseCritic` and `ConditionalPG09`.
The caller provides JSON options, never manually assembles those components.
Source, frozen-base, rows, seed and exact installed core bytes bind the new
cohort. The separate `lifecycle-runtime.lock.json` describes the exact existing
local milestone wheel; it does not update a running environment.

The response contract is **audio / native final AR token hidden state**, shape
`[1, hidden_size]`, continuous vector, neutral student conditioning, no diffusion
time or scheduler, and no waveform/image decoding. Metadata explicitly records
this modality and response kind. Scale zero retains the existing exact bypass;
only strengths in `[0,1]` are supported, with teachers at `+1`. Negative strengths,
reference images, Hub downloads, custom game knobs and native rendering are
rejected. Dummy mode is CPU FP32, native CPU base is FP32 and native CUDA base is
BF16; adapters, bank and critic remain FP32. CUDA work needs separate admission.

The current core still names its registry and packets `image_backends` and
`image-session-v1`. This draft uses that existing protocol without converting any
AR data into image geometry, advertises zero image output types, and records the
modality gap for a common fix. The `evaluate()` result is a hidden response;
it must never be presented as a waveform. `operation='render'` fails before
factory allocation. The legacy native song renderer remains separate and keeps
CFG1, cot off, AR-only slider placement and NAR/VAE scale zero. It is not wired
into this factory until native protocol/listening qualification and the compact
inference seam are available.

After the parent schedules build/install/CPU resource slots, the intended new
cohort API is:

```python
from particle_sliders.lifecycle import SessionConfig, open_session

config = SessionConfig('yue2', 'new-yue2-cpu-cohort', dummy=True,
    rows=[dict(neutral='a quiet acoustic song', positive='a loud guitar song',
               lyrics='[verse]\nA train rolls through the night'),
          dict(neutral='a soft piano song', positive='a bright brass song',
               lyrics='[verse]\nWe meet beside the river')],
    steps=4, seed=7, batch=2, sample_seeds=1, history_tokens=4, ema_decay=.9)
with open_session(config) as session:
    session.train(stop_after=2)  # on_step accepts a HyperGAN-owned observer
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()
    session.export('new-run/adapter.pt', selection='ema')
with open_session(config, artifact='new-run/adapter.pt', operation='evaluate') as session:
    hidden_response = session.evaluate(index=0, seed=11, strength=.5)
```

These are proposed commands, not execution receipts. Install only owner-provided
exact core/PG wheels in a new environment, with no source PYTHONPATH. The new
product wheel has not been built; no product wheel hash is claimed. The factory
checks core wheel SHA256
`9637aede473939e4284cf05c4c8d950e5a84ab5b0fe3c482080eb10028abc4d9` and official
PG0.9.1 wheel identity from the retained lock. Native width2048 on release/auto
remains rejected before pretrained loading. The separately reviewed explicit
continuous profile has an additional current common lifecycle preflight blocker,
detailed in the audit; do not edit released PG bytes or bypass the guard.

Legacy `scripts/train_yue2.py`, `scripts/infer_yue2.py` and Comfy schema paths
still select the old CDF lock and active-weight semantics. Source package naming
is now `yue2_sliders`, following PR3's independent namespace fix; native runtime
imports remain `yue2`. Original product source bytes are checkpoint-bound, so use
the original PR4 checkout for old resumes. The new package metadata deliberately
does not expose legacy installed console scripts, whose original configuration
files remain repository-relative. No old packet is rebound to this lifecycle.
