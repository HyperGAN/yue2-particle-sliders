"""The current YuE2 entry points and its archived guides stay discoverable."""

from pathlib import Path
import re

from yue2_sliders.response_runtime import CORE_PIN, PG_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_links_the_current_response_api():
    readme = _text("README.md")
    for needle in (
        "yue2-particle-sliders",
        "m-a-p/YuE2-3B",
        "m-a-p/YuE2-Vae",
        "ntc-ai/yue2-concept-sliders",
        "yue2_sliders.lifecycle",
        "LIFECYCLE.md",
        CORE_PIN,
        PG_VERSION,
    ):
        assert needle in readme


def test_formulation_keeps_the_shared_api_boundary():
    text = _text("FORMULATION.md")
    for needle in (
        "particle-sliders-core",
        "winning_formulation()",
        "stamp.require",
        "anneal-routed-particle-error-yue2-v1",
        "analysis/slider2d/yue2_gmix_v2_exam.py",
        "RoutedMLP",
        CORE_PIN,
        PG_VERSION,
    ):
        assert needle in text


def test_comfy_and_standalone_commands_have_an_immutable_checkout():
    comfy = _text("COMFYUI.md")
    reproduce = _text("REPRODUCE.md")
    checkout = re.search(r"git checkout --detach ([0-9a-f]{40})", reproduce).group(1)
    archive_url = f"https://github.com/HyperGAN/yue2-particle-sliders/blob/{checkout}"
    assert f"{archive_url}/REPRODUCE.md" in reproduce
    assert f"{archive_url}/COMFYUI.md" in comfy
    assert f"{archive_url}/docs/hypergan-hndl-cohort.md" in reproduce
    assert "NTC/YuE2" in comfy
    assert PG_VERSION in reproduce
