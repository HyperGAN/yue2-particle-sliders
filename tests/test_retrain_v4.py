"""Retrain v4 configs: same as v3 except the opt-in hold04 stamp."""
from pathlib import Path

import yaml

from yue2_sliders.train import parse_args, resolve_surface, _resolve_path
from yue2_sliders.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
HOLD04 = "particle-gmix-1600-hold04-v1"


def _cfg(version, genre):
    return yaml.safe_load((ROOT / f"configs/yue2/retrain-{version}/config-yue2-{genre}-{version}.yaml").read_text())


def test_v4_matches_v3_except_stamp_and_name():
    for genre in ("metal", "house"):
        v3, v4 = _cfg("v3", genre), _cfg("v4", genre)
        assert v4.pop("stamp") == HOLD04
        assert v4.pop("name") == f"{genre}-yue2-particles-v4"
        v3.pop("name")
        assert v4 == v3


def test_v4_resolves_hold04_stamp():
    for genre in ("metal", "house"):
        args = parse_args(["--config_file", str(ROOT / f"configs/yue2/retrain-v4/config-yue2-{genre}-v4.yaml")])
        rows, _ = load_prompts(_resolve_path(Path(args.prompts_file)))
        stamp, locked = resolve_surface(args, len(rows))
        assert stamp.formulation_id == HOLD04
        assert locked["noise_hold_ratio"] == 0.4
        assert locked["noise_decay_steps"] == 1600


def test_v3_still_default_stamp():
    args = parse_args(["--config_file", str(ROOT / "configs/yue2/retrain-v3/config-yue2-metal-v3.yaml")])
    rows, _ = load_prompts(_resolve_path(Path(args.prompts_file)))
    stamp, locked = resolve_surface(args, len(rows))
    assert stamp.formulation_id == "particle-gmix-1600-v2"
    assert locked["noise_hold_ratio"] == 1.3
