# Reproduce

CPU smoke does not download YuE2 weights.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train_yue2.py --help
python scripts/infer_yue2.py --help
python scripts/train_yue2.py --dummy --steps 2 --save_dir /tmp/yue2-smoke
python scripts/infer_yue2.py --dummy \
  --weights /tmp/yue2-smoke/metal-yue2-particles_last.safetensors \
  --output_dir /tmp/yue2-cmp
pytest -q
```

`requirements.txt` pins

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@4340e28bed388d50800c469525b460a108091da0#subdirectory=packages/particle-sliders-core
```

`--dummy` trains the tiny AR stand-in and writes a checkpoint marked `dummy`. Inference reloads it and records hidden-state deltas. Those files are refused for song generation and by the Comfy node.

A live run needs the official runtime in its own environment and a local copy of `m-a-p/YuE2-3B` (plus `m-a-p/YuE2-Vae` to render). Upstream code pin: `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`.

```bash
python -m pip install \
  'yue2-infer @ git+https://github.com/multimodal-art-projection/YuE.git@ef1936f2ee39fe8de486a0f47a481c95f8d4da87'
python scripts/train_yue2.py \
  --config_file configs/yue2/config-yue2-metal.yaml \
  --save_dir models/metal-yue2-particles --allow_hub
```

`--allow_hub` is the opt-in for a download. Without it, loading is local-only. The trainer calls `winning_formulation()` and will not accept a config that restates formulation keys. Prompt sheets for the sixteen published controls are under `configs/yue2/catalog/`.
