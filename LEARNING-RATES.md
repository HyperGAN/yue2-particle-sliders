# Learning rates

## Measured PG0.9.5 native recipe

For the source-qualified native2048 BF16 cohort, use **G = 0.0001**, **D = 0.018**, and **particle bank = 0.030000000000000002** with a common **128-update** schedule. Set `lr=0.0001`, `d_lr_mult=179.99999999999997`, and `prior_lr_mult=300.0`; those exact binary64 products matter. The [options preset](configs/yue2/lr/pg095-native-h128.json) is a subset of common `SessionConfig` options, not a complete session or a new default.

Three G values completed matched32 and matched64 with identical actual D/P, native initialization, teachers and horizon. G0.0001 won the matched64 comparison, between G0.00001 and G0.0003. Its development RAW relative SSE was **0.7767620677498525 at64** and **1.3220490203092785 at128**. The same128 checkpoint then scored **0.5525904019655886** on two previously sealed held-out examples, with no optimizer updates, two valid examples and no exclusions. The above-OFF development128 result is retained; no candidate or endpoint was dropped for that score. [Sanitized evidence](evidence/lr-pg095-native-h128-20261105.json) records all matched scores, exact rates, native mechanical checks and the independent saved-field FP64 audit.

This measurement binds product `2991dfe262b878ae3e5ba1ccd9786c51792c36f2`, core `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, official PG0.9.5, `release-0.9.5/continuous`, and `yue2-bf16-sdpa-math-strict-pg095-v1`. It uses YuE2-3B revision `14fc6c6f146441b1dd6363fcb2e01e82a6914cb7`, Torch2.10.0+cu128, Transformers4.57.6 and yue2-infer0.1.6; rank8/alpha8 q/k/v/o adapters, the complete128×4 bank, batch2, seed7, one sample seed, four history tokens and EMA0.9. RAW, rather than EMA, selects and confirms the rate. This small fixed cohort supports a source-qualified starting recipe; it does not establish a universal optimum or audio/semantic quality.

Configure the [strict native process policy](docs/native-continuous-candidate.md) before model construction, then create a **fresh** session with your paired training rows and local cache:

```python
import json
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

with open('configs/yue2/lr/pg095-native-h128.json') as file:
    options = json.load(file)['options']
config = SessionConfig(
    'yue2', 'my-fresh-yue2-pg095-cohort', **options,
    rows=my_paired_training_rows, device='cuda:0', cache_dir=my_local_cache,
)
with open_response(config) as session:
    session.train()
    session.checkpoint('training.pt')
    session.export_inference('raw.pt', selection='raw')
```

Use the full128 horizon even when inspecting an earlier completed budget. A different source, data, precision, geometry or horizon needs its own evaluation. Existing PG094/PG801 artifacts keep their original environments and are not resumed or relabelled with this recipe. The packaged defaults below remain unchanged.

## Controls and inherited defaults

The current training configuration accepts optional `lr` (base/generator), `d_lr_mult` (critic), and `prior_lr_mult` (actual routed particle bank). Values must be finite positive numbers; booleans are rejected before model allocation. The existing shared ParticleGAN recipe creates all optimizers: D = `lr * d_lr_mult`, bank = `lr * prior_lr_mult`. No product GAN or optimizer math is duplicated.

Omitted controls retain the default `bispf_optimized` recipe: G **0.012**, D multiplier **1.5**, bank multiplier **2.5**. These are inherited defaults, not native calibration results. Selecting the supported `bispf` preset retains its existing base 0.00425, D multiplier 1, bank multiplier 2. Use the current common session options for PG0.9.5 learning rates. The source helper scripts and `yue2_sliders.train` remain historical CDF core0.6/PG0.9.1 producers and retain their original byte-identical interface; their old knobs are not current LR controls.

Use `"lr": 0.00002, "d_lr_mult": 1.5, "prior_lr_mult": 2.5` in common session `options`. The numbers are an input example, not a tuned recommendation. The actual resolved shared recipe and configuration bind full-state resume; a changed rate requires a fresh experiment output/cohort. Older packets and closed HNDL source cohorts keep their original source/defaults. Current CPU optimizer/resume tests do not qualify pretrained/native learning rates, audio or semantic quality.

Native/wide-response source gates remain enforced. Native calibration compares identical initialized state, data, stochastic streams, precision, declared schedule horizons and completed-update budgets on the actual pretrained model. Select and independently confirm rates using only held-out raw teacher/student relative SSE: cast student, teacher and neutral to FP64 before subtraction, pool positions within each example, divide by that example's teacher-edit energy, then average examples equally. Keep the teacher-only valid/excluded set fixed across candidates; zero-energy examples remain undefined, and student failures never remove examples. See the [common LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md).

Preservation/no-edit drift, late/weak-field scores and native media outputs are optional diagnostics; they cannot select, reject or delay a rate. After the native mechanical proof, candidate endpoint net displacement is diagnostic too. Retain exact source/model/data/numerical identity, actual optimizer-rate, finite native-update, checkpoint-continuation and raw-response replay checks. Tiny CPU software checks are not native LR evidence.

The current root package selects core0.13/PG0.9.5. Historical PG094/PG801 measurements remain bound to their original source and are not current0.9.5 recommendations. Keep source/cohort/model/data/numerics and full schedule horizon fixed; old packets cannot accept new source/rate fields on resume.
