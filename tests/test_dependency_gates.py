"""Fail before model loading for incompatible source/domain/config inputs."""
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


def test_full_native_width_rejected_before_pretrained_forward(tmp_path, monkeypatch):
    from yue2 import backend
    from yue2 import train as training

    model_dir = tmp_path / "native-configuration"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"hidden_size":2048}')
    storage = ModuleType("yue2.storage")
    storage.resolve_model = lambda *_args, **_kwargs: model_dir
    monkeypatch.setitem(sys.modules, "yue2.storage", storage)
    monkeypatch.setattr(backend, "require_yue2", lambda: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("pretrained tensor loader reached")
    monkeypatch.setattr(training, "YuE2Backend", forbidden)
    args = training.parse_args(["--steps", "1", "--save_dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="width >=170"):
        training.train(args)
    assert not (tmp_path / "out").exists()
    assert not torch.cuda.is_initialized()


def test_source_probe_rejects_an_already_imported_foreign_core(monkeypatch):
    from yue2.integration import configure_integration

    existing = sys.modules.get("particle_sliders")
    fake = ModuleType("particle_sliders")
    fake.__file__ = "/unqualified/particle_sliders/__init__.py"
    monkeypatch.setitem(sys.modules, "particle_sliders", fake)
    with pytest.raises(ValueError, match="unqualified source"):
        configure_integration()
    assert sys.modules["particle_sliders"] is fake
    if existing is not None:
        monkeypatch.setitem(sys.modules, "particle_sliders", existing)


def test_missing_config_is_not_silently_ignored(tmp_path):
    from yue2.train import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--config_file", str(tmp_path / "missing.yaml"), "--dummy"])


def test_comfy_bootstraps_exact_shared_sources_without_core_pythonpath(monkeypatch):
    import json
    import os
    import subprocess

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    code = """
import json, torch
from comfy_yue2 import YuE2ParticleSlider
handle = YuE2ParticleSlider().load('', 0)[0]
import particle_sliders
print(json.dumps(dict(handle=handle, core=particle_sliders.__file__, cuda=torch.cuda.is_initialized())))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    expected = Path(environment["YUE2_CORE_SOURCE"]) / "packages/particle-sliders-core/src/particle_sliders/__init__.py"
    assert Path(receipt["core"]).resolve() == expected.resolve()
    assert receipt["handle"]["strength"] == 0
    assert receipt["cuda"] is False
