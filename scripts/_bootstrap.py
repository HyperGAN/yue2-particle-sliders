"""Expose source product modules without mutating the native YuE2 namespace."""
from pathlib import Path
import sys


def bootstrap():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.append(root)
