# Formulation

`yue2-particle-sliders` (this repo) owns the **YuE2 product surface**:

- Hub ids: `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`, project `ntc-ai/yue2-concept-sliders`
- the upstream runtime pin `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`
- Comfy gate `comfy_yue2.py` (category `NTC/YuE2`)
- YuE2 AR train and infer (`yue2/train.py`, `yue2/infer.py`, `scripts/train_yue2.py`, `scripts/infer_yue2.py`)
- prompt cards, run configs, and sample cards under `configs/yue2/`

The **game** is not owned here. Train and infer call `winning_formulation()` from `particle-sliders-core` and then `stamp.require(...)`.

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@a119ca1ecd3d5d6c437065839d22739b04f2f4d8#subdirectory=packages/particle-sliders-core
```

That pin is the only shared core. Products do not vendor `RoutedMLP`, `GradRegularizer`, `locked_shared`, or ParticleGAN excerpts. Cap, relativistic loss, and particle VIC come through the stamp (`stamp.regularizer()`, `stamp.losses()`, `stamp.bridge()`, `stamp.critic()`, `stamp.noise_std_at()`).

## What the stamp is right now

Architecture is gmix (routed particles, global-mix critic). That structure is not provisional.

Formulation parameters are provisional until ParticleGAN pull request 38 crowns a full live-leaderboard winner (9 trained toys and all 29 bounds). The overlay id on this pin is `particle-gmix-1600-v2`. Its Hub recipe name is `anneal-routed-particle-error-yue2-v1`. `V2_SPEC` and the exam that records it live in particle-sliders at `analysis/slider2d/yue2_gmix_v2_exam.py`. Products call `winning_formulation()`. Do not fork that stamp math here.

A product config may change model-surface fields only: generator, critic, and particle learning rates, `adv_batch`, `sample_seeds`, `history_tokens`, and `seedbank_sources`. `stamp.require()` rejects a local restatement of rank, particle count, noise horizon, auxiliary weights, or any other formulation key.

## What stays in particle-sliders

Formulation crown docs, 2D gates, and toys stay in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders) (and the conceptmod tree there). That includes `analysis/slider2d/yue2_*_exam.py`, the gmix v2 scoreboard, GAN toy audits, and the C9 native trial. Historical UNI16, Arm B, and `gan_plus_neu` trainers are those experiments. This repo does not reimplement them and does not delete them from particle-sliders.

If a YuE2 run needs a new game, the experiment lands in particle-sliders, the reusable primitive lands in the core (or ParticleGAN), and this product only bumps the commit pin. Copying the knobs into this repo is a fork: `require()` is there to fail that.
