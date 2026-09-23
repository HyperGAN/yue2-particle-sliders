#!/usr/bin/env python3
"""Compare YuE2 slider strengths from this repo.

    python scripts/infer_yue2.py --help
    python scripts/infer_yue2.py --dummy --weights models/run/name_last.safetensors --output_dir /tmp/yue2-cmp

Does not look for a particle-sliders checkout. --dummy does not render audio.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yue2.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
