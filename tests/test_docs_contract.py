"""README and FORMULATION.md keep the product / core split."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_is_the_yue2_product():
    readme = _text("README.md")
    for needle in (
        "yue2-particle-sliders",
        "m-a-p/YuE2-3B",
        "m-a-p/YuE2-Vae",
        "ntc-ai/yue2-concept-sliders",
        "scripts/train_yue2.py",
        "scripts/infer_yue2.py",
        "winning_formulation()",
        "4340e28bed388d50800c469525b460a108091da0",
        "anneal-routed-particle-error-yue2-v1",
        "--dummy",
        "comfy_yue2.py",
    ):
        assert needle in readme, needle
    assert "PARTICLE_SLIDERS_ROOT" not in readme
    assert "does not vendor" in readme


def test_formulation_boundaries():
    text = _text("FORMULATION.md")
    for needle in (
        "yue2-particle-sliders",
        "particle-sliders-core",
        "winning_formulation()",
        "stamp.require",
        "m-a-p/YuE2-3B",
        "anneal-routed-particle-error-yue2-v1",
        "analysis/slider2d/yue2_gmix_v2_exam.py",
        "RoutedMLP",
        "GradRegularizer",
        "ParticleGAN",
        "4340e28bed388d50800c469525b460a108091da0",
    ):
        assert needle in text, needle
    assert "do not fork" in text.lower() or "Do not fork" in text


def test_comfy_and_reproduce_stay_in_repo():
    comfy = _text("COMFYUI.md")
    reproduce = _text("REPRODUCE.md")
    assert "comfy_yue2.py" in comfy
    assert "NTC/YuE2" in comfy
    assert "does not vendor" in comfy
    assert "scripts/infer_yue2.py" in comfy
    assert "scripts/train_yue2.py" in reproduce
    assert "--dummy" in reproduce
    assert "4340e28bed388d50800c469525b460a108091da0" in reproduce
    assert "PARTICLE_SLIDERS_ROOT" not in reproduce


def test_requirements_pin_the_core_once():
    text = _text("requirements.txt")
    pin = "git+https://github.com/HyperGAN/particle-sliders.git@4340e28bed388d50800c469525b460a108091da0#subdirectory=packages/particle-sliders-core"
    assert pin in text
    assert "conceptmod" not in text
