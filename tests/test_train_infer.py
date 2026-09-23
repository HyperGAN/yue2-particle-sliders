"""CPU train/infer smoke. No Hub download and no formulation fork."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from yue2.defaults import CORE_PIN, HUB_RECIPE_NAME, MODEL_ID, PRODUCT_FORMAT
from yue2.infer import infer, parse_args as infer_args
from yue2.slider import ParticleSlider
from yue2.stamp import locked_stamp
from yue2.train import parse_args, train

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_mentions_the_product_model_and_not_a_checkout():
    for script in ("train_yue2.py", "infer_yue2.py"):
        proc = _run(script, ["--help"])
        assert proc.returncode == 0, proc.stderr
        assert MODEL_ID in proc.stdout
        assert "--dummy" in proc.stdout
        assert "winning_formulation" in proc.stdout
        assert "PARTICLE_SLIDERS_ROOT" not in proc.stdout
    for name in ("train_yue2.py", "infer_yue2.py"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "conceptmod.textsliders" not in source
        assert "PARTICLE_SLIDERS_ROOT" not in source


def test_product_sources_do_not_vendor_the_core():
    banned = (
        "class RoutedMLP",
        "class GradRegularizer",
        "class GradientPenalty",
        "def locked_shared",
        "V2_SPEC",
        "PARTICLE_SLIDERS_ROOT",
    )
    roots = [ROOT / "yue2", ROOT / "scripts", ROOT / "comfy_yue2.py"]
    files = []
    for path in roots:
        files.extend(path.rglob("*.py") if path.is_dir() else [path])
    blob = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for needle in banned:
        assert needle not in blob, needle
    assert blob.count("winning_formulation") >= 2


def test_print_card_locks_the_provisional_recipe():
    card = train(parse_args(["--print_card"]))
    assert card["model_id"] == MODEL_ID
    assert card["formulation_provisional"] is True
    assert card["architecture_id"] == "gmix"
    assert card["recipe_name"] == HUB_RECIPE_NAME
    assert card["core_pin"] == CORE_PIN
    assert card["formulation_id"] == "particle-gmix-1600-v2"
    infer_card = infer(infer_args(["--print_card"]))
    assert infer_card["recipe_name"] == card["recipe_name"]
    stamp, locked = locked_stamp()
    stamp.require(locked)
    forked = dict(locked)
    forked["parts"] = 4
    with pytest.raises(ValueError, match="drift"):
        stamp.require(forked)


def test_config_cannot_restate_formulation_keys(tmp_path: Path):
    path = tmp_path / "fork.yaml"
    path.write_text("parts: 4\nname: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formulation"):
        parse_args(["--config_file", str(path)])


def test_dummy_train_and_infer_round_trip(tmp_path: Path):
    sidecar = train(
        parse_args(
            [
                "--dummy",
                "--config_file",
                str(ROOT / "configs" / "yue2" / "config-yue2-metal.yaml"),
                "--name",
                "metal-yue2-dummy",
                "--save_dir",
                str(tmp_path / "run"),
                "--steps",
                "2",
                "--seed",
                "7",
            ]
        )
    )
    payload = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    assert payload["dummy"] is True
    assert payload["allow_hub"] is False
    assert payload["recipe_name"] == HUB_RECIPE_NAME
    assert payload["model_surface"]["sample_seeds"] == 1
    assert payload["model_surface"]["history_tokens"] == 4
    lines = [
        json.loads(line)
        for line in (tmp_path / "run" / "metal-yue2-dummy_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    assert all(math_isfinite(row["g_loss"]) and math_isfinite(row["d_loss"]) for row in lines)
    weights = tmp_path / "run" / "metal-yue2-dummy_last.safetensors"
    comparison = infer(
        infer_args(
            [
                "--dummy",
                "--weights",
                str(weights),
                "--output_dir",
                str(tmp_path / "cmp"),
                "--scales",
                "0,1",
                "--style",
                "English, dry close vocal, soft room.",
            ]
        )
    )
    assert comparison["audio"] is False
    assert comparison["loaded_format"] == PRODUCT_FORMAT
    off = comparison["scales"][0]
    on = comparison["scales"][1]
    assert off["scale"] == 0.0 and on["scale"] == 1.0
    assert off["delta_from_off"] == 0.0
    with pytest.raises(FileExistsError):
        infer(
            infer_args(
                [
                    "--dummy",
                    "--weights",
                    str(weights),
                    "--output_dir",
                    str(tmp_path / "cmp"),
                ]
            )
        )


def test_scale_zero_is_exact_and_dummy_export_is_refused_for_songs(tmp_path: Path):
    sidecar = train(
        parse_args(
            [
                "--dummy",
                "--name",
                "scale-zero",
                "--prompts_file",
                str(ROOT / "configs" / "yue2" / "prompts-yue2.yaml"),
                "--save_dir",
                str(tmp_path / "run"),
                "--steps",
                "1",
                "--config_file",
                str(ROOT / "configs" / "yue2" / "config-yue2.yaml"),
            ]
        )
    )
    del sidecar
    from yue2.backend import TinyYuE2

    weights = tmp_path / "run" / "scale-zero_last.safetensors"
    stamp, _locked = locked_stamp()
    model = TinyYuE2()
    fresh = TinyYuE2()
    fresh.load_state_dict(model.state_dict())
    network, record = ParticleSlider.load(model, weights, stamp, allow_dummy=True)
    assert record["format"] == PRODUCT_FORMAT
    bare = TinyYuE2()
    bare.load_state_dict(fresh.state_dict())
    ids = [3, 4, 5, 6, 7, 8, 9, 10]
    with torch.no_grad():
        for adapter in network.adapters.values():
            adapter.lora_up.weight.add_(0.05)
        baseline = bare.forward_hidden(torch.tensor([ids]))
        with network.scaled(0):
            hooked = model.forward_hidden(torch.tensor([ids]))
        with network.scaled(1):
            moved = model.forward_hidden(torch.tensor([ids]))
    assert torch.equal(baseline, hooked)
    assert not torch.equal(moved, hooked)
    with pytest.raises(ValueError, match="Dummy checkpoints"):
        ParticleSlider.load(TinyYuE2(), weights, stamp)
    assert moved.shape == hooked.shape


def test_penalty_step_stays_finite(tmp_path: Path):
    train(
        parse_args(
            [
                "--dummy",
                "--name",
                "cap-step",
                "--prompts_file",
                str(ROOT / "configs" / "yue2" / "prompts-yue2-metal-arm-b.yaml"),
                "--save_dir",
                str(tmp_path / "run"),
                "--steps",
                "4",
                "--config_file",
                str(ROOT / "configs" / "yue2" / "config-yue2-metal.yaml"),
            ]
        )
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "cap-step_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert rows[-1]["penalty_applied"] is True
    assert math_isfinite(rows[-1]["penalty"])


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
