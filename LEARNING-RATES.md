# Learning rates

## Current finished-audio decision: Crisp Piano RAW64

Use the existing **cell1 G0.00001 RAW64** checkpoint, best of the three saved
rates on four matched cases at batch2/horizon128. Fixed actual D is `0.018`
and bank `0.030000000000000002`; exact prospective common overrides are
`lr=0.00001`, `d_lr_mult=1799.9999999999998`, `prior_lr_mult=3000.0`,
`ema_decay=None`. The equal-case estimated-piano attack-rise ON−OFF mean is
**+0.014077682560999588**. G0.0001 scores`-0.0006603240009776523` and G0.0003
scores`-0.011665111312551407`; keep both negative results. All four selected
case gains and candidate scores are retained in the
[portable decision metadata](docs/finished-output-lr-decision.json), with exact
result/RAW hashes and no private filesystem paths or weights.

This is the **frozen estimated-piano attack-rise proxy**, not human listening or
a general audio-quality score. Experimental stem separation can leak other
instruments and artifacts; two reference anchors establish direction only,
not independent piano attribution in mixtures. This fixed saved RAW64/h128
comparison does not establish a global/joint optimum or change defaults.
Historical training included EMA0.9, but this selection and inference use RAW;
that original identity is retained, while all prospective sessions set EMA None.

## Selection policy and actual common options

The sole score is the equal-case mean intended-effect intensity ON minus
matched OFF in finished outputs. Keep negative/zero outcomes and every case's
weight; technical failures remain missing. Teacher/student relative SSE,
preservation drift, generic beauty, confidence and completion are neither
selectors nor tiebreaks. Reuse completed searches, saved RAW checkpoints,
finished audio and first scores. No new search, generation, rerating or extra
confirmation is required for this decision.

The current response API accepts `lr`, `d_lr_mult`, `prior_lr_mult` and
`ema_decay=None` in `SessionConfig.options`. Shared ParticleGAN derives
D=`lr*d_lr_mult` and bank=`lr*prior_lr_mult`. Preserve the exact literal
multipliers in the linked JSON; rounded quotients can change the binary rates.
The summary is documentation metadata, not a full session or artifact loader.
Omitted controls still use inherited optimized defaults G0.012, D multiplier1.5,
bank multiplier2.5; selecting the supported `bispf` preset retains its own
inherited values. No preset, code, lock or default is changed by this guidance.

For configuration inspection with the real registered API:

```python
import json
from pathlib import Path
from particle_sliders.lifecycle import SessionConfig

decision = json.loads(Path("docs/finished-output-lr-decision.json").read_text())
# supplied_native_options contains the existing paired rows, verified local
# model assets/manifest/cache and exact admitted native numerical/source profile.
options = dict(supplied_native_options)
options.update(decision["rate_overrides"], steps=128, batch=2, seed=7)
config = SessionConfig("yue2", "caller-declared-configuration", **options)
```

This constructs a configuration only, without opening a model session. `steps`
sets the128-update schedule horizon; it is not replaced by the earlier measured
RAW endpoint. Use the already saved selected RAW artifact, rather than retraining
it. Exact checkpoint continuation keeps its original complete configuration,
model/data/source identity and RNG state; these values cannot rebind a different
checkpoint. New sessions explicitly disable model-weight EMA and use RAW weights
throughout; no checkpoint averaging or EMA comparison is proposed.

The historical `yue2_sliders.train` and source helper scripts use their old
archived standalone interface. Their knobs are not the current response-session
rate API. Numerical hidden responses are not waveform generation; the saved
finished-audio comparison uses its own original native consumer.

## Earlier response-error screen

The [archived guide](https://github.com/HyperGAN/yue2-particle-sliders/blob/108d127625c77bae952fc7c957bce3d6f5d48569/LEARNING-RATES.md) retains the original
native preset, all three screened rates and the response-SSE evidence.
