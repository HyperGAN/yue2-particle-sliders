"""Product data in the wheel or the explicit repository source layout.

Setuptools maps the original configs/yue2 directory into _assets without
rewriting or maintaining a second copy of historical locks and prompt cards.
This locates data only; it never selects shared code or dependency cohorts.
"""
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
CONFIGS = _PACKAGE / '_assets'
if not CONFIGS.is_dir():
    CONFIGS = _PACKAGE.parent / 'configs' / 'yue2'
