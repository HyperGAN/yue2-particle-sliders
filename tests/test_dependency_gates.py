"""Installed-wheel identity and fail-closed native domain/config checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_full_native_width_rejected_before_pretrained_forward(tmp_path, monkeypatch):
    from yue2_sliders import backend
    from yue2_sliders import train as training

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


def test_installed_package_probe_rejects_an_already_imported_foreign_core(monkeypatch):
    from yue2_sliders.integration import configure_integration

    existing = sys.modules.get("particle_sliders")
    fake = ModuleType("particle_sliders")
    fake.__file__ = "/unqualified/particle_sliders/__init__.py"
    monkeypatch.setitem(sys.modules, "particle_sliders", fake)
    with pytest.raises((ValueError, RuntimeError), match="origin|unqualified|installed"):
        configure_integration()
    assert sys.modules["particle_sliders"] is fake
    if existing is not None:
        monkeypatch.setitem(sys.modules, "particle_sliders", existing)


@pytest.mark.parametrize("name", ["particle-sliders-core", "particlegan"])
def test_installed_wheel_version_mismatch_is_rejected(monkeypatch, name):
    from yue2_sliders import integration

    original = integration.metadata.distribution
    monkeypatch.setattr(integration.metadata, "distribution", lambda package:
                        SimpleNamespace(version="0.10.0") if package == name else original(package))
    with pytest.raises(ValueError, match="version differs"):
        integration.configure_integration()
    assert not torch.cuda.is_initialized()


def test_modified_installed_package_bytes_are_rejected_without_editing_them(monkeypatch):
    from importlib.metadata import distribution
    from yue2_sliders.integration import configure_integration

    target = Path(distribution("particle-sliders-core").locate_file("particle_sliders/pg09_audio.py")).resolve()
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path:
                        b"unqualified adapter source" if path.resolve() == target else original(path))
    with pytest.raises(ValueError, match="sources differ"):
        configure_integration()
    assert not torch.cuda.is_initialized()


def test_missing_config_is_not_silently_ignored(tmp_path):
    from yue2_sliders.train import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--config_file", str(tmp_path / "missing.yaml"), "--dummy"])


def test_installed_package_and_comfy_work_without_checkout_paths():
    from importlib.metadata import distribution
    from yue2_sliders.integration import configure_integration

    dependency = configure_integration()
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "YUE2_AUDIO_SOURCE", "YUE2_CORE_SOURCE"):
        environment.pop(name, None)
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    code = """
import json, sys, torch
from yue2_sliders.integration import configure_integration
dependency = configure_integration()
from comfy_yue2 import YuE2ParticleSlider
handle = YuE2ParticleSlider().load('', 0)[0]
import particle_sliders
from particle_sliders.pg09_audio import YuE2ARResponseAdapter
from particle_sliders.pg09_backends import AdapterParameters
print(json.dumps(dict(handle=handle, dependency=dependency, core=particle_sliders.__file__,
    audio_module=YuE2ARResponseAdapter.__module__, backend_module=AdapterParameters.__module__,
    executable=sys.executable, legacy_audio_imported=any(n.startswith('conceptmod.textsliders') for n in sys.modules),
    cuda=torch.cuda.is_initialized())))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    expected = distribution("particle-sliders-core").locate_file("particle_sliders/__init__.py")
    assert Path(receipt["core"]).resolve() == expected.resolve()
    assert Path(receipt["executable"]).resolve() == Path(sys.executable).resolve()
    assert receipt["dependency"] == dependency
    assert receipt["dependency"]["core"]["version"] == "0.6.0"
    assert receipt["dependency"]["particlegan"]["version"] == "0.9.1"
    assert receipt["audio_module"] == "particle_sliders.pg09_audio"
    assert receipt["backend_module"] == "particle_sliders.pg09_backends"
    assert receipt["legacy_audio_imported"] is False
    assert receipt["handle"]["strength"] == 0
    assert receipt["cuda"] is False
