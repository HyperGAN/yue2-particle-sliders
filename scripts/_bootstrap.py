"""Bind product-only modules to the installed YuE2 namespace, when available."""

from __future__ import annotations

import importlib
from importlib.machinery import PathFinder
from pathlib import Path
import sys


def bootstrap():
    root = Path(__file__).resolve().parents[1]
    original = list(sys.path)
    # Never let the product's yue2/__init__.py hide the native runtime. This
    # includes the empty entry when a CLI is invoked from the product root.
    sys.path[:] = [entry for entry in original if Path(entry or '.').resolve() != root]
    try:
        spec = PathFinder.find_spec('yue2', sys.path)
        existing = sys.modules.get('yue2')
        if existing is not None and Path(existing.__file__).resolve().is_relative_to(root):
            if spec is None:
                return existing  # Repeated dummy/Comfy bootstrap without a native install.
            raise RuntimeError('Import the native YuE2 runtime before product modules')
        runtime = importlib.import_module('yue2') if spec is not None else None
    finally:
        sys.path[:] = original
    if runtime is None:
        if str(root) not in sys.path:
            sys.path.append(str(root))
        return importlib.import_module('yue2')
    if Path(runtime.__file__).resolve().is_relative_to(root):
        raise RuntimeError('The product yue2 package was imported before the native CLI bootstrap')
    product = root / 'yue2'
    if str(product) in runtime.__path__:
        return runtime
    for source in product.glob('*.py'):
        if source.name == '__init__.py':
            continue
        if any((Path(directory) / source.name).exists() for directory in runtime.__path__):
            raise RuntimeError('Native/product submodule collision: ' + source.name)
    if str(product) not in runtime.__path__:
        runtime.__path__.append(str(product))
    return runtime
