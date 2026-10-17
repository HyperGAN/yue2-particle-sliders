# These commands ran in the isolated derivative checkout on 2026-10-17.
# Read-only existing environments; no installs, CUDA jobs or service changes.
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export YUE2_AUDIO_SOURCE=/mnt/nvme/sliders/audio-adapters-pg09-20261017
export YUE2_CORE_SOURCE=/mnt/nvme/sliders/audio-core-pin-86fc4e0-20261017
export PYTHONPATH="$YUE2_CORE_SOURCE/packages/particle-sliders-core/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/mnt/nvme/sliders/qualification/envs/pg090/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/yue2-final-frozen-cpu --junitxml=evidence/pg09-derivative/full-cpu.xml
# Native environment, import/CLI help only. No PG source overlay is needed or
# claimed by this package/class import test; it creates no model or tensors.
env -u PYTHONPATH /ml2/sliders/.venv-yue2-retrain/bin/python scripts/train_yue2.py --help
env -u PYTHONPATH /ml2/sliders/.venv-yue2-retrain/bin/python scripts/infer_yue2.py --help
# Native import receipt: scripts._bootstrap.bootstrap();
# from yue2.backend import require_yue2; require_yue2();
# inspect source paths, direct_url.json and torch.cuda.is_initialized().
