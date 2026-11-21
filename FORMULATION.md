# YuE2 response formulation

`yue2-particle-sliders` owns YuE2 token packing, prompt cards, and AR adapter wiring for `m-a-p/YuE2-3B`. `particle-sliders-core` is pinned at `ec1161f5d74481be3540cb0dfa63d0e8f8b54e90`. The installed runtime verifies its complete source lock and the official ParticleGAN 0.11.1 source before opening a response session.

`winning_formulation()` and `stamp.require(...)` bind the routed adapter geometry and the Hub stamp `anneal-routed-particle-error-yue2-v1`. The bridge uses shared `RoutedMLP`. The shared engine owns the game and ParticleGAN supplies the public recipe's loss, penalty, prior regularizer and optimizer. The historical exam lives in `analysis/slider2d/yue2_gmix_v2_exam.py`.

Each frozen positive teacher and neutral student shares a sampled AR history. Conditioning retains the full neutral hidden state with sequence and history lengths and sign. The routed adapter uses one 128×4 bank and scale zero bypasses its updates. Native hidden responses keep their 2048 coordinates and require explicit `release-0.11.1` with `continuous` input-domain selection. The shared default automatic domain retains its width gate.

The [response lifecycle](LIFECYCLE.md) covers fresh RAW-only runs and exact FULL continuation. The [reproduction guide](REPRODUCE.md) identifies earlier standalone and published-weight cohorts.
