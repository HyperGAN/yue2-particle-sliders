"""The repo package must not shadow the official YuE2 runtime package `yue2`."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_repo_has_no_yue2_package():
    assert not (ROOT / "yue2" / "__init__.py").exists()
    assert (ROOT / "yue2_sliders" / "__init__.py").exists()


def test_runtime_module_path_resolves_outside_repo():
    spec = importlib.util.find_spec("yue2")
    if spec is None:
        pytest.skip("yue2-infer runtime not installed")
    assert ROOT not in Path(spec.origin).resolve().parents
    pytest.importorskip("yue2.modeling_yue2")
    from yue2_sliders.backend import require_yue2

    config_cls, model_cls = require_yue2()
    assert config_cls.__module__ == "yue2.modeling_yue2"
