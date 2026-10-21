# YuE2 sliders

The product uses the single shared `ConditionalPG09` engine with public PG0.9
GAN-only recipes. See [FORMULATION.md](../FORMULATION.md) for conditions,
objectives, actual routed-bank ownership, source/domain and checkpoint contracts,
and [REPRODUCE.md](../REPRODUCE.md) for exact installed wheel dependencies.

AR attention `model.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}` trains.
NAR, frozen AR base weights and VAE remain unchanged. The positive teacher and
neutral student share AR history. Only +1 is trained; 0 to 1 probes are supported.
Negative strength remains unqualified. Scale 0 is the frozen base exactly.

`winning_formulation()` supplies routed bridge geometry only. Public PG0.9
presets own optimizer rates, active losses and prior regularization. Configs
select prompt sheet, total steps, batch, history and seed-bank size. Local
formulation overrides and legacy learning-rate knobs fail closed.

The full training checkpoint contains shared game/optimizer/sampler/RNG state.
Resume retains the recipe horizon and checks base, sources, conditions and rows
before writing outputs. Inference weights are active adapter and bank weights,
not EMA. Base-model identity and stamp/tensor schema are checked before hooks.
Dummy inference reconstructs the exact original seeded base, computes the off
baseline independently of requested scale order, and writes JSON without audio.

Live render uses the installed native eager runtime, same neutral style,
lyrics and seed at each strength, CFG1/cot-off, and AR-only scaling. NAR and
VAE run at scale0. Real render/acoustic qualification remains pending; a CPU
hidden-state response is not acoustic evidence. Historical published samples
remain indexed in `configs/yue2/sample-cards.yaml`.
