# Fresh new environment; existing native/PG090/candidate42 environments preserved.
/mnt/nvme/sliders/qualification/envs/pg090/bin/python -m venv --system-site-packages \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-final
/ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-final/bin/python -m pip install \
  --no-cache-dir --no-index --no-deps \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-14/pg091-release/particlegan-0.9.1-py3-none-any.whl \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-14/releases/4be0b5b29d2e2c741431f468d71ba8a7c884b724/dist/particle_sliders_core-0.6.0-py3-none-any.whl
# Shared core imports must resolve installed site-packages; no checkout PYTHONPATH.
env -u PYTHONPATH -u YUE2_AUDIO_SOURCE -u YUE2_CORE_SOURCE \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  YUE2_PG091_CONTINUOUS=/ml2/codex-storage/future-workspaces/2026-10-17/task-11/pg091-continuous \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-final/bin/python \
  -m pytest -q -p no:cacheprovider --basetemp=/tmp/yue2-pg091-final-full \
  --junitxml=evidence/pg091-wheel/final-full.xml
# The optional standalone profile contains only the verified49 PG Python files;
# no shared checkout or foreign site-packages, no installed release edits.
env -u PYTHONPATH -u YUE2_AUDIO_SOURCE -u YUE2_CORE_SOURCE \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-final/bin/python \
  evidence/pg091-wheel/collect_runtime.py > evidence/pg091-wheel/final-runtime.json
# Native preserved environment: help/class imports only; no candidate core install.
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /ml2/sliders/.venv-yue2-retrain/bin/python scripts/train_yue2.py --help
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /ml2/sliders/.venv-yue2-retrain/bin/python scripts/infer_yue2.py --help
