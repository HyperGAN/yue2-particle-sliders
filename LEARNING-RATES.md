# Learning rates

The current training configuration accepts optional `lr` (base/generator), `d_lr_mult` (critic), and `prior_lr_mult` (actual routed particle bank). Values must be finite positive numbers; booleans are rejected before model allocation. The existing shared ParticleGAN recipe creates all optimizers: D = `lr * d_lr_mult`, bank = `lr * prior_lr_mult`. No product GAN or optimizer math is duplicated.

Omitted controls retain the default `bispf_optimized` recipe: G **0.012**, D multiplier **1.5**, bank multiplier **2.5**. These are inherited defaults, not native calibration results. Selecting the supported `bispf` preset retains its existing base 0.00425, D multiplier 1, bank multiplier 2. Use the current common session options for PG0.9.4 learning rates. The source helper scripts and `yue2_sliders.train` remain historical CDF core0.6/PG0.9.1 producers and retain their original byte-identical interface; their old knobs are not current LR controls.

Use `"lr": 0.00002, "d_lr_mult": 1.5, "prior_lr_mult": 2.5` in common session `options`. The numbers are an input example, not a tuned recommendation. The actual resolved shared recipe and configuration bind full-state resume; a changed rate requires a fresh experiment output/cohort. Older packets and closed HNDL source cohorts keep their original source/defaults. Current CPU optimizer/resume tests do not qualify pretrained/native learning rates, audio or semantic quality.

Native/wide-response source gates remain enforced. Native calibration needs the exact pretrained model, an admitted upstream domain/API source, identical fixed data/seeds/update budgets, held-out edit and preservation evaluation, and durable source/recipe/results evidence.
