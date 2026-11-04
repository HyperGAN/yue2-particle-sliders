# YuE2 native continuous source candidate

The public response interface preserves native `[1,2048]` AR states. The default
`release-0.9.4/auto` profile still refuses this width before loading the model.
A separate explicit candidate binds core
`5d68473a098801ee1ab3ebbb12463daef51c6346` and ParticleGAN source
`801ae1ead8f88b88eca8f5398137e3e1bc01ad36`, using
`pg_source_profile="source-801ae1e-continuous-v1"` and
`penalty_input_domain="continuous"`. Install the exact candidate PG wheel/source
in a fresh environment; the official PyPI 0.9.4 wheel is a different source and
is rejected for this candidate. For an immutable source installation:

```sh
python -m pip install "particlegan @ git+https://github.com/255BITS/ParticleGAN.git@801ae1ead8f88b88eca8f5398137e3e1bc01ad36"
python -m pip install ".[native]"
```

The shared core independently checks every imported ParticleGAN Python source
against its qualified manifest. Use a fresh environment, not an archived song
or control reproduction environment.

The native loader uses `yue2-infer` 0.1.6 from immutable upstream commit
`ef1936f2ee39fe8de486a0f47a481c95f8d4da87`. Its actual dependency requirements
include Torch 2.10.0 and Transformers 4.57.6. This candidate additionally binds
actual Torch `2.10.0+cu128` / CUDA build `12.8`, BF16 base precision and an explicit
CUDA device. Metadata may report Torch `2.10.0`; the runtime build is checked
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
    revision='14fc6c6f146441b1dd6363fcb2e01e82a6914cb7',
    cache_dir='/path/to/local/huggingface/hub',
    pg_source_profile='source-801ae1e-continuous-v1',
    penalty_input_domain='continuous',
)
```

This is preparation for a separately admitted real-model proof. CPU random
fixtures validate the common optimizer, exact checkpoint replay and raw/EMA
native-only export contract. Source/asset verification and a complete meta
loader key/shape comparison validate readiness without a pretrained forward.
They do not establish native training, acoustic quality or a recommended LR.
A measured real-model probe must verify teacher/OFF responses, both optimizer
updates, exact resume, frozen base and portable inference before an LR search.
