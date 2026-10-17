# Reproduce

This draft uses installed `particle-sliders-core` **0.6.0** candidate commit
`4be0b5b29d2e2c741431f468d71ba8a7c884b724` and released PG **0.9.1**.
`configs/yue2/shared-runtime.lock.json` binds the exact wheel hashes and installed
shared Python/JSON bytes. The candidate is local and is not claimed as a public
release. No guessed Git URL, shared checkout or source path variable is used.

Create a new environment; keep existing native/runtime cohorts untouched. Install
these exact wheels (provided by the package owner), then remaining dependencies:

```bash
python -m venv /path/to/new-environment
/path/to/new-environment/bin/python -m pip install --no-deps "$PG_WHEEL" "$CORE_WHEEL"
/path/to/new-environment/bin/python -m pip install -r requirements.txt
```

| Artifact | SHA256 |
|---|---|
| `particle_sliders_core-0.6.0-py3-none-any.whl` | `2e7621c370e790f65702acf76cfa2f428b393cd3f863245aa26d8a76ff096ada` |
| `particlegan-0.9.1-py3-none-any.whl` | `b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec` |

The runtime probe rejects different wheel identity, edited shared bytes, or a
shared module imported from a checkout. It never mutates `sys.path`, installs a
package or falls back to another branch. Old `--audio_source`/`--core_source`
flags are rejected. Tests and commands run with no shared checkout PYTHONPATH:

```bash
unset PYTHONPATH YUE2_CORE_SOURCE YUE2_AUDIO_SOURCE
export CUDA_VISIBLE_DEVICES=''
python scripts/train_yue2.py --help
python scripts/infer_yue2.py --help
python scripts/train_yue2.py --dummy --steps 3 --until 1 --save_dir /tmp/yue2-first
python scripts/train_yue2.py --dummy --steps 3 \
  --resume /tmp/yue2-first/metal-yue2-particles_checkpoint.pt --save_dir /tmp/yue2-resumed
python scripts/infer_yue2.py --dummy \
  --weights /tmp/yue2-resumed/metal-yue2-particles_last.safetensors \
  --output_dir /tmp/yue2-cmp
python -m pytest -q
```

`--steps` is the fixed recipe horizon; `--until` pauses without changing it.
Resume uses the full `_checkpoint.pt` and a fresh or empty output directory.
`_adapter-export.pt` is the shared inference payload, and `_last.safetensors`
contains active product weights. Dummy checkpoints contain base seed/identity
and are refused for songs and Comfy. All comparisons use identical base weights.

The public PG recipe owns updates; the host binds FP32 training explicitly.
Installed class names, source hashes, PG version, model/row/condition identities
and optimizer/RNG state are checkpoint-bound. This is a **new cohort** relative
to historical source-based PG 0.9.0. Old full checkpoints are rejected, and no
active old checkpoint or environment is migrated automatically.

Native runtime pin remains `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`
(`yue2-infer`). CLI imports installed native first, then adds product modules.
Native dependencies stay native; never inject a foreign environment's entire
site-packages. This wheel-only CPU environment does not qualify a combined native
GPU runtime or acoustic behavior.

Native width 2048 remains rejected by release/auto before pretrained loading.
The separately staged exact **PG 0.9.1** continuous profile, with explicit
`--pg_source_profile alz-continuous-v1 --penalty_input_domain continuous`, is
required for wider continuous vectors. It is neither applied nor selected by the
product. Optional synthetic width2048 tests take `YUE2_PG091_CONTINUOUS` pointing
at the separately verified 49-source profile; they do not use shared checkout
code or pretrained models. No installed release bytes are edited.

GPU work still requires coordinator admission and new native qualification.
Y20 continuation, accepted exports, native render/listening and longer work
remain separate audio-owner gates. Historical published samples do not qualify
this PG 0.9.1 trainer's acoustic quality. Ada reviews exact heads before merge;
isolated development, installation, CPU testing and draft updates precede that
review.
