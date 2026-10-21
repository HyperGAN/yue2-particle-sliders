# Narrow CDF follow-up only. Prior4be104/full and32/gates receipts remain unchanged.
/mnt/nvme/sliders/qualification/envs/pg090/bin/python -m venv --system-site-packages \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-cdf
/ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-cdf/bin/python -m pip install \
  --no-cache-dir --no-index --no-deps \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-14/pg091-release/particlegan-0.9.1-py3-none-any.whl \
  /ml2/codex-storage/future-workspaces/2026-10-17/task-14/releases/cdfb80a76b882b4b73861b12f555263bd64ffeb7/dist/particle_sliders_core-0.6.0-py3-none-any.whl
# Both commands used this clean environment (set exports in the new shell):
unset PYTHONPATH YUE2_AUDIO_SOURCE YUE2_CORE_SOURCE
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-cdf/bin/python -m pytest -q \
  tests/test_pg091_wheel_gates.py -k clock -p no:cacheprovider \
  --basetemp=/tmp/yue2-cdf-clocks --junitxml=evidence/pg091-cdf/clocks.xml
/ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-cdf/bin/python -m pytest -q \
  tests/test_dependency_gates.py::test_installed_package_and_comfy_work_without_checkout_paths \
  tests/test_pg09_training.py::test_entry_point_executes_shared_game_with_live_bank_and_gan_only \
  tests/test_pg09_training.py::test_cli_fresh_process_resume_matches_uninterrupted_game_and_active_export \
  tests/test_pg091_wheel_gates.py::test_initial_empty_optimizer_history_remains_valid_product_resume \
  tests/test_pg091_wheel_gates.py::test_honest_unused_critic_history_remains_valid_product_resume \
  tests/test_train_infer.py::test_dummy_train_and_infer_round_trip \
  tests/test_inference_regressions.py::test_reload_uses_exported_actual_bank_and_adapter_sign \
  -p no:cacheprovider --basetemp=/tmp/yue2-cdf-consumer \
  --junitxml=evidence/pg091-cdf/consumer.xml
/ml2/codex-storage/future-workspaces/2026-10-17/task-11/venv-pg091-cdf/bin/python \
  evidence/pg091-wheel/collect_runtime.py > evidence/pg091-cdf/runtime.json
