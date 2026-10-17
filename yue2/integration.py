"""Verify the exact installed candidate wheel; never add shared source paths.

The local candidate is not a published release. Its bytes are pinned in the
product lock. Installing or selecting packages belongs to environment setup.
"""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys

LOCK_PATH = Path(__file__).resolve().parents[1] / "configs/yue2/shared-runtime.lock.json"
PG_VERSION = "0.9.1"
CORE_PIN = "cdfb80a76b882b4b73861b12f555263bd64ffeb7"


def _distribution(record):
    distribution = metadata.distribution(record["name"])
    if distribution.version != record["version"]:
        raise ValueError(f"{record['name']} version differs from the pinned wheel")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    hashes = direct.get("archive_info", {}).get("hashes", {})
    if hashes.get("sha256") != record["wheel_sha256"]:
        raise ValueError(f"{record['name']} wheel identity differs; install the exact locked wheel")
    return distribution


def configure_integration():
    """Verify installed bytes and origins before importing the shared API.

    PG's separately declared domain profile remains validated by the shared
    pre-forward gate. No checkout discovery, path mutation or package fallback.
    """
    lock = json.loads(LOCK_PATH.read_text())
    core, pg = lock["core"], lock["particlegan"]
    if core["commit"] != CORE_PIN or pg["version"] != PG_VERSION:
        raise ValueError("Product runtime lock and code pins disagree")
    distribution = _distribution(core)
    _distribution(pg)
    root = Path(distribution.locate_file("")).resolve()
    package = root / "particle_sliders"
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob("*")) if p.is_file() and p.suffix in (".py", ".json")}
    if actual != core["sources"]:
        raise ValueError("Installed shared package sources differ from the pinned wheel")
    for name, module in tuple(sys.modules.items()):
        if name == "particle_sliders" or name.startswith("particle_sliders."):
            source = getattr(module, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(package):
                raise ValueError(f"already imported {name} from an unqualified source: {source}")
    for name in ("particle_sliders", "particle_sliders.pg09", "particle_sliders.pg09_audio",
                 "particle_sliders.pg09_backends"):
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        relative = str(source.relative_to(root)) if source.is_relative_to(root) else None
        if relative not in actual:
            raise ValueError(f"import origin differs from the installed candidate wheel: {name}")
    return dict(core=dict(core, path=str(package)), particlegan=pg, cohort=lock["cohort"])
