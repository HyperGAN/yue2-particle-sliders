#!/usr/bin/env python3
"""Train a YuE2 routed-particle slider from this repo.

    python scripts/train_yue2.py --help
    python scripts/train_yue2.py --dummy

The game is particle_sliders.winning_formulation(). This script does not
look for a particle-sliders checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yue2.train import main  # noqa: E402


if __name__ == "__main__":
    main()
