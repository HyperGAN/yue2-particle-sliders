"""Sample-card index for the published YuE2 listening gallery."""

from __future__ import annotations

from pathlib import Path

import yaml

from yue2.defaults import HUB_PROJECT

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "configs" / "yue2" / "sample-cards.yaml"
CATALOG = ROOT / "configs" / "yue2" / "catalog"


def load_sample_cards(path: Path | None = None) -> dict:
    document = yaml.safe_load((path or CARDS).read_text(encoding="utf-8"))
    if document.get("hub") != HUB_PROJECT:
        raise ValueError(f"Sample cards must use the {HUB_PROJECT} Hub project")
    if not document.get("cards"):
        raise ValueError("Sample cards are empty")
    return document


def hub_resolve(path: str, hub: str | None = None) -> str:
    return f"https://huggingface.co/{hub or HUB_PROJECT}/resolve/main/{path}"


def featured_links(document: dict | None = None) -> list[dict]:
    document = document or load_sample_cards()
    hub = document["hub"]
    links = []
    for card in document["cards"]:
        links.append(
            {
                "id": card["id"],
                "title": card["title"],
                "prompts_file": card["prompts_file"],
                "seed": document["seed"],
                "strength": document["strength"],
                "off": hub_resolve(card["off"], hub),
                "on": hub_resolve(card["on"], hub),
            }
        )
    return links


def catalog_controls() -> list[str]:
    """Sixteen published controls, taken from the train prompt filenames."""
    names = sorted(path.name[: -len("-train.yaml")] for path in CATALOG.glob("*-train.yaml"))
    if len(names) != 16:
        raise ValueError(f"Expected 16 catalog controls, found {len(names)}")
    return names
