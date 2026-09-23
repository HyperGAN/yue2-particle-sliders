"""Prompt cards, sample index, and the model lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yue2.defaults import CORE_PIN, HUB_PROJECT, HUB_RECIPE_NAME, MODEL_ID, VAE_ID
from yue2.prompts import load_prompts, sound_only
from yue2.samples import catalog_controls, featured_links, load_sample_cards

ROOT = Path(__file__).resolve().parents[1]


def test_model_lock_names_the_product_and_core_pin():
    lock = json.loads((ROOT / "configs" / "yue2" / "model.lock.json").read_text(encoding="utf-8"))
    assert lock["model_id"] == MODEL_ID
    assert lock["vae_id"] == VAE_ID
    assert lock["hub_project"] == HUB_PROJECT
    assert lock["hub_recipe_name"] == HUB_RECIPE_NAME
    assert lock["core"]["pin"] == CORE_PIN
    assert "yue2_gmix_v2_exam.py" in lock["formulation_note"]
    assert "winning_formulation" in lock["formulation_note"]


def test_sound_only_rejects_named_references():
    with pytest.raises(ValueError, match="named references"):
        sound_only("a ballad in the style of a famous singer")
    sound_only("close breathy vocal, small dry room, fingerpicked steel strings")


def test_prompt_cards_are_unipolar_and_sound_only():
    paths = list((ROOT / "configs" / "yue2").rglob("prompts-*.yaml"))
    paths += list((ROOT / "configs" / "yue2" / "catalog").glob("*.yaml"))
    assert len(paths) >= 16
    for path in paths:
        rows, meta = load_prompts(path)
        assert rows
        assert meta.get("recommended_range", [0, 1]) == [0, 1]
        for row in rows:
            assert row["neutral"] != row["positive"]
            assert "negative" not in row


def test_catalog_has_sixteen_controls_and_featured_audio():
    controls = catalog_controls()
    assert controls == sorted(controls)
    assert {"female", "metal", "house", "hiphop"} <= set(controls)
    document = load_sample_cards()
    links = featured_links(document)
    assert [item["id"] for item in links] == ["female", "metal", "house"]
    assert all(item["off"].startswith(f"https://huggingface.co/{HUB_PROJECT}/") for item in links)
    assert all(item["seed"] == 1709 and item["strength"] == 1 for item in links)
    for item in links:
        assert (ROOT / item["prompts_file"]).is_file()
