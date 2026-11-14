# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Native AR teacher/hidden preparation | 1 | 12 | 24 | one sequence ≤ 256 total prompt+history tokens; BF16/math SDPA; full checkpoint |
| Native AR response training | 1 | 14 | 24 | batch 2; ≤ 256 total tokens/row; shared block checkpointing; BF16 base/FP32 adapters |
| Separate eager finished-audio pipeline: AR/NAR then VAE | 1 | 92 | 28 | batch 1; semantic generation ≤ 9000 tokens, total AR context/latent frames ≤ 24576; BF16 model + FP32 VAE; math-attention upper envelope, no AR offload |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.9.5 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **m-a-p/YuE2-3B**, revision **`14fc6c6f146441b1dd6363fcb2e01e82a6914cb7`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| Native AR teacher/hidden preparation | frozen: 6.7627; prefill: 0.0273; workspace: 4.0000 |
| Native AR response training | frozen: 6.7627; trainable_transient: 0.2386; checkpoint_activations: 0.3320; cache: 0.0078; workspace: 6.0000 |
| Separate eager finished-audio pipeline: AR/NAR then VAE | model: 6.7627; VAE: 0.4940; adapter: 0.0149; largest_attention_pair: 72.0000; KV_full_context_upper: 2.6250; linear_prefill: 1.8750; workspace: 8.0000 |

Detailed arithmetic and assumptions:

- **BF16 model bytes**: `7261368448`.
- **VAE FP32 bytes**: `530464520`.
- **LoRA elements**: `3211264`.
- **adapted projections**: `112`.
- **generator plus critic elements**: `4002721`.
- **KV formula**: `2*28*1*8*24576*128*2`.
- **decode**: `FP32 VAE core1024frames+16-frame halos; output bound24576*1920 stereo48k frames; host output bytes=377487360`.
- **host loading**: `full BF16 CPU model + one full7,261,441,640-byte checkpoint loading buffer + FP32 VAE +8 GiB +trainable export copies`.

Scope and limits:

- The core numerical response API refuses waveform rendering; infer.py is a separate official eager pipeline and keeps NAR/VAE frozen/OFF. The whole model parameter total already includes its checkpoint branches; do not add a fictitious second3B model.
- The20 GiB CLI memory_budget_gib is a scheduler setting, not the calculation. Long full-context math attention can require much more. Short songs or a verified memory-efficient attention implementation can use less; that is not a measured claim. Unbounded lyric/prompt/latent requests have no finite universal maximum.
- No quantized model or AR-offload profile is accepted by the current live renderer. Old EMA history is not an enabled prospective profile; use RAW/no EMA.

Source facts (read as text; no model execution):

- [yue2_sliders/native_assets.py](yue2_sliders/native_assets.py)
- [yue2_sliders/native-assets.lock.json](yue2_sliders/native-assets.lock.json)
- [yue2_sliders/lifecycle.py](yue2_sliders/lifecycle.py)
- [yue2_sliders/backend.py](yue2_sliders/backend.py)
- [yue2_sliders/infer.py](yue2_sliders/infer.py)
- [configs/yue2/model.lock.json](configs/yue2/model.lock.json)
- [Pinned model metadata](https://huggingface.co/m-a-p/YuE2-3B/tree/14fc6c6f146441b1dd6363fcb2e01e82a6914cb7) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.9.5 release `1b8355b184b21bb7d81104ee590813c262f6df0a`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `pg09_audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
