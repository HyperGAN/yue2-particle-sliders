"""Retrain v3 configs parse and match the v2 run budget."""
from pathlib import Path
import pytest
from yue2.train import parse_args, resolve_surface, _resolve_path
from yue2.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("genre", ["metal", "house"])
def test_v3_config(genre):
    args = parse_args(["--config_file", str(ROOT / f"configs/yue2/retrain-v3/config-yue2-{genre}-v3.yaml")])
    rows, _ = load_prompts(_resolve_path(args.prompts_file))
    assert args.steps == 1600 and len(rows) == 8
    _, locked = resolve_surface(args, len(rows))
    assert locked["seedbank_sources"] == 512
    assert any("Genre: electronic song" in row["neutral"] for row in rows)
