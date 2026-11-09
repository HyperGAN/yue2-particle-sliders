# YuE2 released PG0.9.5 native continuous cohort

The public response interface preserves native `[1,2048]` states. Core0.13.0 at `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423` uses published PG0.9.5. Select `pg_source_profile="release-0.9.5"`, `penalty_input_domain="continuous"`; auto retains its width gate. Do not install the old PG801 overlay into this environment.

```sh
python -m pip install "particlegan==0.9.5"
python -m pip install ".[native]"
```

The complete released PG source manifest is checked independently. This is a fresh numerical/source cohort: old PG094/PG801 packets need their original environment. CPU fixtures do not qualify new pretrained execution or transfer old LR results.

The native loader uses `yue2-infer` 0.1.6 from immutable upstream commit
`ef1936f2ee39fe8de486a0f47a481c95f8d4da87`. Its actual dependency requirements
include Torch 2.10.0 and Transformers 4.57.6. This candidate additionally binds
actual Torch `2.10.0+cu128` / CUDA build `12.8`, BF16 base precision and an explicit
CUDA device. The caller must explicitly select
`native_numerical_profile="yue2-bf16-sdpa-math-strict-pg095-v1"` and configure its process
before model construction. The product only inspects these settings and refuses
a mismatch; it never changes the caller's process policy. The profile requires
erroring deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, highest
FP32 matmul precision, TF32 disabled for matmul/cuDNN, cuDNN deterministic mode
with benchmarking disabled, and FP16/BF16 reduced-precision reductions disabled.
Native attention is explicitly loaded as `sdpa`; its native grouped-query call
to `torch.nn.functional.scaled_dot_product_attention` admits only the math
backend (flash, memory-efficient and cuDNN SDPA disabled, low-precision math
reductions disabled). TF32 force-on environment overrides are refused. The
actual flags and attention implementation become part of native identity and
raw/EMA inference reconstruction. The same identity is checked before each
native optimizer update and observed inference/evaluation operation.

Metadata may report Torch `2.10.0`; the runtime build is checked
separately. The full dependency map is in `native_assets.py`.

Use an existing local `m-a-p/YuE2-3B` snapshot at revision
`14fc6c6f146441b1dd6363fcb2e01e82a6914cb7`, passed as the full `revision` and its
Hugging Face `cache_dir`. The loader never downloads weights. Before loading,
`native-assets.lock.json` verifies every cached model/tokenizer file and every
installed upstream package source file, the upstream immutable Git origin and
the native architecture. After loading, every frozen model tensor must equal
its BF16 checkpoint tensor before an adapter can attach; staged runtimes must
pass the same complete check. The BF16 checkpoint contains 3,630,684,224 parameters
(6.76 GiB of tensor data), before activations or adapters. Hidden-response work
does not load the optional YuE2 VAE.

Training still delegates all GAN losses, regularizers, optimizers, particle
prior and continuation state to shared `ConditionalPG09`. This change preserves
the current public routed adapter bank and shared audio critic. The separate
archived HNDL and historical CDF/PG091 cohorts are unchanged.

A candidate session adds these options to the normal paired sound/lyrics rows:

```python
native_options = dict(
    device='cuda:0', base_precision='bfloat16',
    native_numerical_profile='yue2-bf16-sdpa-math-strict-pg095-v1',
    revision='14fc6c6f146441b1dd6363fcb2e01e82a6914cb7',
    cache_dir='/path/to/local/huggingface/hub',
    pg_source_profile='release-0.9.5',
    penalty_input_domain='continuous',
)
```

The original PG0.9.5 full2/exact1+1, teacher/OFF, frozen-base and public RAW
proofs remain retained. The current [finished-audio decision](../LEARNING-RATES.md)
uses existing cell1 G0.00001 RAW64/h128 and a four-case equal-mean piano-stem
attack-rise ON−OFF proxy. Its separation leakage limits remain explicit; it is
not a broad acoustic-quality claim. The old G0.0001 RAW128 teacher-SSE screen
is diagnostic history, not the current selector. No repeated search, native
proof, render, rerating or extra confirmation is required. CPU fixtures still
have software scope; native assets/numerics and exact checkpoint identity do not
change with this documentation.

The caller must set the strict policy before opening a session or inference
artifact. Set the CUBLAS workspace environment before importing Torch:

```python
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
```

This explicit native policy does not apply to the CPU dummy fixture or alter
the strict default official PG release profile.
# Construction RNG ownership

Native construction seeds the CPU default generator and only the explicitly
requested CUDA device inside a forked RNG scope. Both caller states are restored
on success or failure; other visible CUDA generators are neither seeded nor
queried. The native identity records `cpu-and-requested-cuda-fork-v1`. This is a
new native source cohort, not a rebinding of the earlier strict-source artifacts.
The CPU dummy fixture and archived reproduction sources keep their prior RNG
behavior. This change does not qualify pretrained GPU execution or audio quality.

The native training factory also scopes provider, critic and game construction;
the preserved loader RNG must not leave the critic dependent on caller state.
Source identity includes the factory, provider and history-preparation bytes,
read without importing training modules during native-only inference. Fresh
reconstruction must match actual network/critic/prior/stream/config states before
continuation. The earlier native cbbf/v4 attempt stopped on a critic-initialization
mismatch; its packets are not migrated to this source cohort.
