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
CDF/core0.6/PG0.9.1 interface. Their knobs are not the current response-session
rate API. Numerical hidden responses are not waveform generation; the saved
finished-audio comparison uses its own original native consumer.

<a id="measured-pg095-native-recipe"></a>

## Historical teacher-response SSE screen (G0.0001 RAW128)

The older G0.0001 nominee below was selected by teacher-relative SSE, not by
the current finished-audio score. Its original source, scores, presets and
EMA identity remain diagnostic history. It does not override the existing
G0.00001 RAW64 effect-selected checkpoint or require another confirmation.


For the source-qualified native2048 BF16 cohort, use **G = 0.0001**, **D = 0.018**, and **particle bank = 0.030000000000000002** with a common **128-update** schedule. Set `lr=0.0001`, `d_lr_mult=179.99999999999997`, and `prior_lr_mult=300.0`; those exact binary64 products matter. The [options preset](configs/yue2/lr/pg095-native-h128.json) is a subset of common `SessionConfig` options, not a complete session or a new default.

Three G values completed matched32 and matched64 with identical actual D/P, native initialization, teachers and horizon. G0.0001 won the matched64 comparison, between G0.00001 and G0.0003. Its development RAW relative SSE was **0.7767620677498525 at64** and **1.3220490203092785 at128**. The same128 checkpoint then scored **0.5525904019655886** on two previously sealed held-out examples, with no optimizer updates, two valid examples and no exclusions. The above-OFF development128 result is retained; no candidate or endpoint was dropped for that score. [Sanitized evidence](evidence/lr-pg095-native-h128-20261105.json) records all matched scores, exact rates, native mechanical checks and the independent saved-field FP64 audit.

This measurement binds product `2991dfe262b878ae3e5ba1ccd9786c51792c36f2`, core `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, official PG0.9.5, `release-0.9.5/continuous`, and `yue2-bf16-sdpa-math-strict-pg095-v1`. It uses YuE2-3B revision `14fc6c6f146441b1dd6363fcb2e01e82a6914cb7`, Torch2.10.0+cu128, Transformers4.57.6 and yue2-infer0.1.6; rank8/alpha8 q/k/v/o adapters, the complete128×4 bank, batch2, seed7, one sample seed, four history tokens. That measurement maintained an EMA0.9 shadow, but every
selection and confirmation value used RAW; its evidence and packets are unchanged. This small fixed cohort supports a source-qualified starting recipe; it does not establish a universal optimum or audio/semantic quality.

The original EMA0.9 shadow remains part of that historical training identity,
while its selected/current exported values are RAW. Preserve the old packet and
configuration exactly; do not retroactively label that training no-EMA or resume
it with new `ema_decay=None`. New sessions and inference use RAW/no new weight
EMA. Historical PG094/PG801 packets keep their original environments.
