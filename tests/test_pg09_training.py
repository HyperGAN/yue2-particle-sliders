"""CPU entry-point checks using the installed shared package and PG 0.9.1.

No source checkout paths, pretrained models, native audio runtime, or CUDA
device are required.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            equal_tree(first, second)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


@pytest.fixture
def installed_package(monkeypatch):
    from yue2_sliders.integration import configure_integration

    for name in ("PYTHONPATH", "YUE2_AUDIO_SOURCE", "YUE2_CORE_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    dependency = configure_integration()
    assert dependency["core"]["version"] == "0.6.0"
    assert dependency["core"]["wheel_sha256"]
    assert dependency["core"]["sources"]
    assert dependency["particlegan"]["version"] == "0.9.1"
    assert dependency["particlegan"]["wheel_sha256"]
    return dependency


def arguments(tmp_path, *, name="pg09-cpu", steps=3, extra=()):
    return [
        "--dummy", "--name", name, "--save_dir", str(tmp_path),
        "--steps", str(steps), "--save_every", "1", "--seed", "7",
        *extra,
    ]


def checkpoint(directory, name="pg09-cpu"):
    return torch.load(directory / f"{name}_checkpoint.pt", map_location="cpu", weights_only=True)


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_entry_point_executes_shared_game_with_live_bank_and_gan_only(tmp_path, installed_package, monkeypatch, preset):
    from particle_sliders import ConditionalPG09
    from particlegan import get_recipe
    from yue2_sliders.train import parse_args, train

    original = ConditionalPG09.step
    observed = {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("MSE/reconstruction objective reached")

    monkeypatch.setattr(torch.nn.functional, "mse_loss", forbidden)

    def traced(game):
        adapter = game.backend
        owned = {id(p) for p in game.network.parameters()}
        assert adapter.particles is adapter.slider.particles
        assert adapter.particles.shape == (128, 4)
        assert sum(p is adapter.particles for group in game.opt_g.param_groups for p in group["params"]) == 1
        if not observed:
            observed.update(
                calls=0,
                before=deepcopy(game.network.state_dict()),
                base={name: p.detach().clone() for name, p in adapter.model.named_parameters()
                      if id(p) not in owned},
            )
        for prepared in game.rows:
            assert prepared.row.scale == 1.0
            assert torch.equal(adapter.sample(prepared, scale=0), prepared.neutral)
        state = game.state_dict()
        rng = torch.get_rng_state().clone()
        game.evaluate(0, seed=19)
        assert torch.equal(torch.get_rng_state(), rng)
        equal_tree(game.state_dict(), state)
        metrics = original(game)
        assert metrics["g_loss"] == pytest.approx(metrics["g_adv"] + metrics["prior_regularization"])
        assert metrics["d_loss"] == pytest.approx(metrics["d_adv"] + metrics["d_pen"])
        assert asdict(game.recipe) == asdict(get_recipe(
            preset, total_steps=3, batch_size=2, num_particles=128, z_dim=4,
            training_precision="float32",
        ))
        assert all(torch.equal(observed["base"][name], p) for name, p in adapter.model.named_parameters()
                   if name in observed["base"])
        observed["calls"] += 1
        observed["after"] = deepcopy(game.network.state_dict())
        observed["game"] = game
        return metrics

    monkeypatch.setattr(ConditionalPG09, "step", traced)
    sidecar = train(parse_args(arguments(tmp_path, extra=("--preset", preset))))
    assert Path(sidecar).is_file()
    assert observed["calls"] == 3
    assert all(not torch.equal(value, observed["after"][name])
               for name, value in observed["before"].items())
    game = observed["game"]
    assert int(game.exposures["d"].sum()) == 3 * game.batch_size
    assert int(game.exposures["g"].sum()) == 3 * game.batch_size
    packet = checkpoint(tmp_path)
    equal_tree(packet["game"], game.state_dict())
    assert packet["frozen_base_sha256"]
    # Public product inference weights must contain the active trained state.
    equal_tree(load_file(str(tmp_path / "pg09-cpu_last.safetensors")), game.backend.slider.state_dict())
    assert not torch.cuda.is_initialized()


def run_cli(args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    for name in ("PYTHONPATH", "YUE2_AUDIO_SOURCE", "YUE2_CORE_SOURCE"):
        env.pop(name, None)
    return subprocess.run([sys.executable, str(ROOT / "scripts/train_yue2.py"), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)


def test_real_cli_preserves_explicit_dummy_steps(tmp_path, installed_package):
    flags = arguments(tmp_path, steps=4)
    offset = flags.index("--steps")
    flags[offset:offset + 2] = ["--steps=4"]
    result = run_cli(flags)
    assert result.returncode == 0, result.stdout + result.stderr
    assert checkpoint(tmp_path)["game"]["completed_steps"] == 4
    rows = [json.loads(line) for line in (tmp_path / "pg09-cpu_train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]


def test_cli_fresh_process_resume_matches_uninterrupted_game_and_active_export(tmp_path, installed_package):
    full, first, resumed = [tmp_path / name for name in ("full", "first", "resumed")]
    for destination, extra in (
        (full, ()),
        (first, ("--until", "1")),
        (resumed, ("--resume", str(first / "pg09-cpu_checkpoint.pt"))),
    ):
        result = run_cli(arguments(destination, steps=3, extra=extra))
        assert result.returncode == 0, result.stdout + result.stderr
    a, b = checkpoint(full), checkpoint(resumed)
    assert checkpoint(first)["game"]["completed_steps"] == 1
    assert a["game"]["completed_steps"] == b["game"]["completed_steps"] == 3
    equal_tree(a["game"], b["game"])
    equal_tree(a["metrics"], b["metrics"])
    equal_tree(a["probes"], b["probes"])
    assert a["frozen_base_sha256"] == b["frozen_base_sha256"]
    equal_tree(load_file(str(full / "pg09-cpu_last.safetensors")),
               load_file(str(resumed / "pg09-cpu_last.safetensors")))


def test_resume_identity_failure_does_not_create_output(tmp_path, installed_package):
    from yue2_sliders.train import parse_args, train

    first, rejected = tmp_path / "first", tmp_path / "rejected"
    train(parse_args(arguments(first, extra=("--until", "1"))))
    packet = checkpoint(first)
    packet["game"]["config"]["conditions_sha256"] = "wrong-condition-bytes"
    incompatible = tmp_path / "incompatible.pt"
    torch.save(packet, incompatible)
    original = incompatible.read_bytes()
    with pytest.raises(ValueError, match="mismatch|incompatible|identity"):
        train(parse_args(arguments(rejected, extra=("--resume", str(incompatible)))))
    assert not rejected.exists()
    assert incompatible.read_bytes() == original


def test_training_uses_installed_package_without_checkout_configuration(tmp_path, installed_package):
    from yue2_sliders.train import parse_args

    args = parse_args(["--dummy", "--save_dir", str(tmp_path)])
    equal_tree(args.dependency, installed_package)
    assert not hasattr(args, "audio_source")
    assert not hasattr(args, "core_source")
    for flag in ("--audio_source", "--core_source"):
        with pytest.raises(SystemExit):
            parse_args(["--dummy", flag, str(tmp_path)])
