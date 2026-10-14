"""YuE2 prompt cards. Sound descriptions only; no named-reference syntax."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_NAMED = re.compile(
    r"\b(?:in the style of|inspired by|sounds like)\b|\S\s+[—–]\s+meaning\b",
    re.IGNORECASE,
)


def sound_only(text: str) -> str:
    """Reject named-reference syntax. Prompts must describe the sound itself."""
    if not isinstance(text, str):
        raise ValueError("Prompt text must be a string")
    if _NAMED.search(text):
        raise ValueError("Describe instruments, voice, room and timing; remove named references")
    return text


def load_prompts(path: str | Path) -> tuple[list[dict], dict]:
    """Load a product prompt card.

    Rows need ``neutral``, ``positive`` and ``lyrics``. Optional ``attributes``
    repeat a pair. There is no negative teacher. ``recommended_range`` must be
    the unipolar interval ``[0, 1]`` when present.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"rows": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list) or not raw["rows"]:
        raise ValueError("Prompts need a nonempty rows list")
    allowed = {"rows", "plus_label", "zero_label", "recommended_range"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"Unsupported prompt metadata: {sorted(extra)}")
    recommended = raw.get("recommended_range", [0, 1])
    if list(recommended) != [0, 1]:
        raise ValueError("YuE2 sliders use recommended_range [0, 1]")
    meta = {key: value for key, value in raw.items() if key != "rows"}
    for value in meta.values():
        if isinstance(value, str):
            sound_only(value)
    rows: list[dict] = []
    for item in raw["rows"]:
        if not isinstance(item, dict):
            raise ValueError("Each prompt row must be a mapping")
        if "negative" in item:
            raise ValueError("YuE2 rows are unipolar: no negative teacher")
        base = {}
        for key in ("neutral", "positive", "lyrics"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Each row needs nonempty {key}")
            base[key] = sound_only(value.strip())
        if base["neutral"] == base["positive"]:
            raise ValueError("Neutral and positive styles must differ")
        unknown = set(item) - {"neutral", "positive", "lyrics", "attributes"}
        if unknown:
            raise ValueError(f"Unsupported prompt fields: {sorted(unknown)}")
        attributes = item.get("attributes") or [""]
        if not isinstance(attributes, list) or not all(isinstance(value, str) for value in attributes):
            raise ValueError("attributes must be a list of sound descriptions")
        for attribute in attributes:
            sound_only(attribute)
            prefix = attribute.strip()
            rows.append(
                {
                    "neutral": f"{prefix} {base['neutral']}".strip(),
                    "positive": f"{prefix} {base['positive']}".strip(),
                    "lyrics": base["lyrics"],
                }
            )
    return rows, meta
