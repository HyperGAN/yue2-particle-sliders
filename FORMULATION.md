# Formulation

`yue2-particle-sliders` owns YuE2 token packing, prompt cards, AR adapter wiring,
CLI, checkpoint files and Comfy integration for `m-a-p/YuE2-3B`.

All training updates call the shared `ConditionalPG09` from exact local `particle-sliders-core` source
`3e94b13eb5d6cd1ceee3f018f019a68603f35696`. The existing audio response adapter,
critic, conditions and ownership/checkpoint wrappers come from exact local audio
integration `9262a930cd351459e93ac33cc542fa379c12fe52`. These are internal source
APIs pending a separately reviewed packaged release. Do not fork them here.

The public ParticleGAN **0.9.0** `bispf` or `bispf_optimized` recipe owns D/G
losses, public prior regularization, penalties, optimizers, rate policy, balanced
or explicitly selected replacement sampling, RNG streams and checkpoint restore.
Generator loss is `g_adv + public_prior`; discriminator loss is
`d_adv + public_penalty`. No MSE, reconstruction, distillation, extra VIC term,
EMA shadow, or legacy local update loop participates.

`winning_formulation()` and `stamp.require(...)` retain the routed adapter
architecture and its legacy stamp `anneal-routed-particle-error-yue2-v1`.
`stamp.bridge()` supplies `RoutedMLP`; this product does not vendor that class,
`GradRegularizer`, or ParticleGAN. The old global-mix critic/objective is not the
PG09 training game. Historical formulation exams stay in particle-sliders at
`analysis/slider2d/yue2_gmix_v2_exam.py`.

Each frozen positive teacher and neutral student share identical sampled AR
continuation history. Conditions bind the full frozen neutral last hidden state,
neutral/positive lengths, history length, and +1 sign. The actual 128x4 particle
bank participates in routed generation and is passed once to the shared engine.
The adapter stays AR-only. Scale 0 bypasses every adapter exactly.

Native responses remain `[1,2048]`, without projection or reshaping. The default
release/auto PG source profile rejects continuous vectors of width >=170. The
explicit `alz-continuous-v1`/`continuous` pair requires the exact independently
reviewed 49-source overlay; there is no automatic fallback. Optimizer rates come
from the public recipe; legacy `g_lr`, `d_lr`, and `particle_lr` flags fail closed.

Training checkpoints bind actual base weights, native/runtime/source identities,
recipe, teacher rows, conditions, domain, active network/critic/bank, optimizers,
regularizer, D/G sampler state, private noise/model RNG and completed steps.
Inference files contain the active adapter and bank. They cannot resume training.
Changing package class paths or source hashes starts a new qualified cohort;
old admitted native checkpoints must remain on their original pinned sources.
