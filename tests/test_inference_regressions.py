"""CPU regression coverage for active adapters and fail-closed inference."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from comfy_yue2 import validate_slider_file
from yue2.backend import TinyYuE2, attention_targets
from yue2.defaults import CORE_PIN, MODEL_ID, PUBLISHED_FORMAT
from yue2.infer import _dummy_compare, parse_args
from yue2.integration import AUDIO_PIN
from yue2.slider import ParticleSlider, RoutedProjection, read_record
from yue2.stamp import locked_stamp

ROOT = Path(__file__).resolve().parents[1]
BASE_IDENTITY = {"config_sha256": "cpu-regression-base", "files": {"model.safetensors": "fixed"}}


def _checkpoint(tmp_path, *, dummy=False):
    stamp, _ = locked_stamp()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = TinyYuE2()
        # The stand-in has full-size K/V. This coherent configuration also lets
        # the model-free Comfy validator check the real native shape formula.
        model.config.num_key_value_heads = model.config.num_attention_heads
        base = {key: value.clone() for key, value in model.state_dict().items()}
        network = ParticleSlider(model, stamp)
        with torch.no_grad():
            for adapter in network.adapters.values():
                adapter.lora_up.weight.normal_(std=0.03)
    path = tmp_path / "slider.safetensors"
    metadata = dict(
        core_pin=CORE_PIN, architecture_id=stamp.architecture_id,
        formulation_id=stamp.formulation_id, recipe_name=stamp.spec["recipe_name"],
        model_id=MODEL_ID, model_identity={"dummy": True, "model_id": MODEL_ID} if dummy else BASE_IDENTITY,
        dummy=dummy, seed=7,
    )
    network.save(path, metadata)
    return stamp, path, base


def _rewrite(path, *, updates=None, remove=(), tensors=None):
    record = read_record(path)
    record.update(updates or {})
    for key in remove:
        record.pop(key, None)
    save_file(load_file(str(path)) if tensors is None else tensors, str(path),
              metadata={"yue2_particle_sliders": json.dumps(record)})


@pytest.mark.parametrize("key,value", [
    ("recipe_name", "foreign"), ("core_pin", "foreign"),
    ("formulation", "foreign"), ("formulation_id", "foreign"),
    ("architecture_id", "foreign"), ("rank", 1), ("alpha", 1),
    ("particles", 1), ("particle_dim", 1), ("upstream_revision", "foreign"),
    ("model_id", "foreign"), ("model_identity", {}),
])
def test_bad_provenance_rejected_before_hook_or_base_mutation(tmp_path, key, value):
    stamp, path, _ = _checkpoint(tmp_path)
    _rewrite(path, updates={key: value})
    model = TinyYuE2()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError):
        ParticleSlider.load(model, path, stamp, model_identity=BASE_IDENTITY)
    assert all(not isinstance(getattr(layer.forward, "__self__", None), RoutedProjection)
               for layer in attention_targets(model).values())
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_missing_identity_and_different_base_fail_before_hooks(tmp_path):
    stamp, path, _ = _checkpoint(tmp_path)
    model = TinyYuE2()
    with pytest.raises(ValueError, match="actual base model_identity"):
        ParticleSlider.load(model, path, stamp)
    with pytest.raises(ValueError, match="different base"):
        ParticleSlider.load(model, path, stamp, model_identity={"config_sha256": "another-base"})
    _rewrite(path, remove=("model_identity",))
    with pytest.raises(ValueError, match="model_identity"):
        ParticleSlider.load(model, path, stamp)
    assert all(not isinstance(getattr(layer.forward, "__self__", None), RoutedProjection)
               for layer in attention_targets(model).values())


def test_reload_uses_exported_actual_bank_and_adapter_sign(tmp_path):
    stamp, path, base = _checkpoint(tmp_path)
    model = TinyYuE2()
    model.config.num_key_value_heads = model.config.num_attention_heads
    model.load_state_dict(base)
    ids = torch.tensor([[3, 5, 7, 11]])
    with torch.no_grad():
        bare = model.forward_hidden(ids)
    network, _record = ParticleSlider.load(model, path, stamp, model_identity=BASE_IDENTITY)
    assert torch.equal(network.particles, load_file(str(path))["particles"])
    assert all(adapter._particles is network.particles for adapter in network.adapters.values())
    with torch.no_grad():
        with network.scaled(0):
            assert torch.equal(model.forward_hidden(ids), bare)
        with network.scaled(1):
            assert not torch.equal(model.forward_hidden(ids), bare)
        adapter = next(iter(network.adapters.values()))
        x = torch.randn(2, 4, model.config.hidden_size)
        baseline = adapter.org_forward(x)
        with network.scaled(1):
            positive = adapter(x)
        with network.scaled(-1):
            negative = adapter(x)
        assert torch.allclose(positive - baseline, baseline - negative, atol=1e-7)
    with pytest.raises(RuntimeError):
        with network.scaled(0.5):
            raise RuntimeError("restore after failure")
    assert all(adapter.multiplier == 0 for adapter in network.adapters.values())


def test_comfy_checks_actual_tensor_payload(tmp_path):
    stamp, path, _base = _checkpoint(tmp_path)
    assert validate_slider_file(path, stamp)["path"] == str(path)
    _rewrite(path, tensors={"foreign": torch.ones(1)})
    with pytest.raises(ValueError, match="tensors"):
        validate_slider_file(path, stamp)


def test_valid_published_checkpoint_keeps_legacy_recipe_provenance(tmp_path):
    stamp, path, _base = _checkpoint(tmp_path)
    _rewrite(path, updates={
        "format": PUBLISHED_FORMAT, "recipe": stamp.spec["recipe_name"],
        "bridge_width": int(stamp.spec["adapter_width"]), "router_width": int(stamp.spec["router_width"]),
    }, remove=("recipe_name", "core_pin", "architecture_id", "formulation_id", "formulation"))
    assert validate_slider_file(path, stamp)["format"] == PUBLISHED_FORMAT


@pytest.mark.parametrize("fault", ["missing", "version", "objective", "weights", "base", "source"])
def test_conditional_export_signature_cannot_claim_another_game(tmp_path, fault):
    stamp, path, _base = _checkpoint(tmp_path)
    dependencies = {family: {"commit": commit, "sources": {"source.py": "recorded-sha256"}}
                    for family, commit in (("audio", AUDIO_PIN), ("core", CORE_PIN))}
    signature = dict(
        pg_version="0.9.0", objective="g_adv+public_prior;d_adv+public_penalty",
        response_domain="continuous_vector", generator_arm="routed_particle_gan",
        recipe={"name": "bispf_optimized"},
        backend=dict(family="yue2_native_ar", student_conditioning="neutral", trained_scales=[1.0],
                     model_identity=dict(base=BASE_IDENTITY, integration_dependencies=deepcopy(dependencies))),
    )
    updates = dict(training_game="conditional-pg09-v1", training_recipe="bispf_optimized",
                   training_weights="active", ema=False, integration_dependencies=dependencies, game_signature=signature)
    _rewrite(path, updates=updates)
    assert validate_slider_file(path, stamp)["path"] == str(path)
    if fault == "missing":
        updates["game_signature"] = None
    elif fault == "version":
        signature["pg_version"] = "0.10.0"
    elif fault == "objective":
        signature["objective"] = "g_adv+auxiliary_mse"
    elif fault == "weights":
        updates["training_weights"] = "ema"
    elif fault == "base":
        signature["backend"]["model_identity"]["base"] = {"different": "base"}
    else:
        dependencies["core"]["commit"] = "another-core"
    _rewrite(path, updates=updates)
    with pytest.raises(ValueError, match="Conditional PG09"):
        validate_slider_file(path, stamp)


def test_dummy_delta_uses_off_even_when_off_is_last_or_absent_and_replays(tmp_path):
    stamp, path, _base = _checkpoint(tmp_path, dummy=True)
    args = parse_args(["--dummy", "--weights", str(path), "--output_dir", str(tmp_path / "out"), "--scales", "1,0"])
    rng = torch.get_rng_state().clone()
    first = _dummy_compare(args, stamp)
    assert first == _dummy_compare(args, stamp)
    assert torch.equal(rng, torch.get_rng_state())
    assert first["scales"][0]["delta_from_off"] > 0
    assert first["scales"][1]["delta_from_off"] == 0
    args.scales = [1.0]
    assert _dummy_compare(args, stamp)["scales"][0] == first["scales"][0]


def test_bootstrap_imports_native_first_even_from_product_working_directory(tmp_path):
    native = tmp_path / "native" / "yue2"
    native.mkdir(parents=True)
    (native / "__init__.py").write_text("NATIVE_SENTINEL = True\n", encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(native.parent), str(ROOT / "scripts")]))
    proc = subprocess.run([sys.executable, "-c",
        "from _bootstrap import bootstrap; import json; m = bootstrap(); bootstrap(); "
        "print(json.dumps([m.__file__, m.NATIVE_SENTINEL, list(m.__path__)]))"],
        cwd=ROOT, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    origin, sentinel, paths = json.loads(proc.stdout)
    assert Path(origin) == native / "__init__.py"
    assert sentinel is True
    assert paths == [str(native), str(ROOT / "yue2")]


def test_direct_product_import_discovers_native_submodules(tmp_path):
    native = tmp_path / "native" / "yue2"
    native.mkdir(parents=True)
    (native / "__init__.py").write_text("", encoding="utf-8")
    (native / "protocol.py").write_text("SENTINEL = 17\n", encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(ROOT), str(native.parent)]))
    proc = subprocess.run([sys.executable, "-S", "-c", "from yue2.protocol import SENTINEL; print(SENTINEL)"],
                          cwd=tmp_path, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "17"


@pytest.mark.parametrize("options", [["--scales=-1,0,1"], ["--scales=0,1.5"], ["--cot", "full"], ["--abc_file", "score.abc"]])
def test_cli_refuses_unqualified_polarity_and_symbolic_conditioning(options):
    with pytest.raises(SystemExit):
        parse_args(["--print_card", *options])


def test_comfy_custom_node_import_works_without_product_root_on_sys_path(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")
    source = (
        "import importlib.util, sys; "
        f"s = importlib.util.spec_from_file_location('custom_yue2', {str(ROOT / '__init__.py')!r}, "
        f"submodule_search_locations=[{str(ROOT)!r}]); "
        "m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m); "
        "print(sorted(m.NODE_CLASS_MAPPINGS))"
    )
    proc = subprocess.run([sys.executable, "-c", source], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "NTCYuE2ParticleSlider" in proc.stdout


def test_live_render_freezes_cfg_and_scales_only_ar(monkeypatch):
    from yue2.infer import render_live
    protocol = SimpleNamespace(SongRequest=lambda **kwargs: SimpleNamespace(**kwargs, to_dict=lambda: kwargs))
    result_cls = lambda *args: SimpleNamespace(args=args)
    monkeypatch.setitem(sys.modules, "yue2.protocol", protocol)
    monkeypatch.setitem(sys.modules, "yue2.pipeline", SimpleNamespace(SongResult=result_cls))
    monkeypatch.setitem(sys.modules, "yue2.storage", SimpleNamespace(identity=lambda value: value))
    class Network:
        scale = 0
        @contextmanager
        def scaled(self, scale):
            previous, self.scale = self.scale, scale
            try:
                yield
            finally:
                self.scale = previous
    network = Network()
    calls = []
    class Pipe:
        backend, quantization, offload_ar, weights = "torch-eager", "none", False, {"mot": BASE_IDENTITY}
        def plan(self, request):
            assert request.cfg_scale == 1.0
            calls.append(("plan", network.scale, request.seed))
            return SimpleNamespace(timing={})
        def generate_semantic(self, plan, sampling):
            calls.append(("semantic", network.scale, sampling))
            return SimpleNamespace(timing={})
        def synthesize(self, semantic):
            calls.append(("nar", network.scale))
            return "latents"
        def decode(self, latents):
            calls.append(("vae", network.scale))
            return "audio"
        def effective_config(self, request, semantic_sampling):
            return {}
    result = render_live(Pipe(), network, style="dry piano", lyrics="hello", scale=0.5,
                         seed=19, cot="off", abc=None, semantic_sampling={"max_tokens": 3}, adapter_identity="adapter")
    assert calls == [("plan", 0.5, 19), ("semantic", 0.5, {"max_tokens": 3}), ("nar", 0), ("vae", 0)]
    assert result.args[4]["yue2_particle_sliders"]["scale"] == 0.5
    assert network.scale == 0
