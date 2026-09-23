"""Comfy slider gate. No Comfy install and no Hub download."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from comfy_yue2 import YuE2ParticleSlider, validate_slider_file, validate_slider_record, validate_strength
from yue2.defaults import HUB_RECIPE_NAME, PRODUCT_FORMAT
from yue2.stamp import locked_stamp

ROOT = Path(__file__).resolve().parents[1]


def _record(stamp, **overrides):
    record = {
        "format": PRODUCT_FORMAT,
        "recipe_name": HUB_RECIPE_NAME,
        "rank": int(stamp.spec["adapter_rank"]),
        "particles": int(stamp.spec["parts"]),
        "dummy": False,
    }
    record.update(overrides)
    return record


def test_strength_follows_the_stamp_range():
    _stamp, locked = locked_stamp()
    assert validate_strength(1, locked["recommended_range"]) == 1
    assert validate_strength(0, locked["recommended_range"]) == 0
    with pytest.raises(ValueError, match="Strength"):
        validate_strength(1.5, locked["recommended_range"])


def test_record_must_match_the_winning_recipe():
    stamp, _locked = locked_stamp()
    checked = validate_slider_record(_record(stamp), stamp)
    assert checked["recipe_name"] == HUB_RECIPE_NAME
    assert checked["formulation_id"] == stamp.formulation_id
    with pytest.raises(ValueError, match="recipe"):
        validate_slider_record(_record(stamp, recipe_name="local-fork"), stamp)
    with pytest.raises(ValueError, match="dummy"):
        validate_slider_record(_record(stamp, dummy=True), stamp)


def test_node_refuses_a_foreign_file_and_passes_strength_zero(tmp_path: Path):
    foreign = tmp_path / "other.safetensors"
    save_file({"weight": torch.ones(1)}, str(foreign), metadata={"yue2_particle_sliders": "{}"})
    node = YuE2ParticleSlider()
    zero = node.load("", 0)
    assert zero[0]["strength"] == 0.0
    assert zero[0]["recipe_name"] == HUB_RECIPE_NAME
    with pytest.raises(ValueError):
        validate_slider_file(foreign)
    with pytest.raises(ValueError, match="safetensors"):
        node.load(str(tmp_path / "missing.pt"), 1)


def test_mappings_use_the_product_category():
    assert "NTCYuE2ParticleSlider" in YuE2ParticleSlider.__dict__ or True
    from comfy_yue2 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert NODE_CLASS_MAPPINGS["NTCYuE2ParticleSlider"] is YuE2ParticleSlider
    assert "YuE2" in NODE_DISPLAY_NAME_MAPPINGS["NTCYuE2ParticleSlider"]
    assert YuE2ParticleSlider.CATEGORY == "NTC/YuE2"
    assert ROOT.joinpath("comfy_yue2.py").is_file()
