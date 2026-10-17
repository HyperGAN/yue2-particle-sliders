"""Explicit, immutable source dependencies for the integration-only PG 0.9 CLI.

These internal adapters are not a published wheel API. Never search for a
checkout, install packages, or fall back to another branch here.
"""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

AUDIO_PIN = "9262a930cd351459e93ac33cc542fa379c12fe52"
CORE_PIN = "3e94b13eb5d6cd1ceee3f018f019a68603f35696"


def _source(root, pin, files):
    if not root:
        raise ValueError("PG09 integration requires explicit --audio_source and --core_source checkouts (or YUE2_AUDIO_SOURCE/YUE2_CORE_SOURCE)")
    root = Path(root).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != pin:
        raise ValueError(f"integration source {root} is {head}; required {pin}")
    digests = {}
    for relative in files:
        path = root / relative
        committed = subprocess.check_output(["git", "-C", str(root), "show", f"{pin}:{relative}"])
        if path.read_bytes() != committed:
            raise ValueError(f"integration source differs from pinned commit: {path}")
        digests[relative] = hashlib.sha256(committed).hexdigest()
    return root, dict(path=str(root), commit=head, sources=digests)


def configure_integration(audio_source=None, core_source=None):
    """Verify exact source bytes before importing the sole shared update engine."""
    audio, audio_id = _source(audio_source or os.environ.get("YUE2_AUDIO_SOURCE"), AUDIO_PIN, (
        "conceptmod/__init__.py", "conceptmod/textsliders/__init__.py",
        "conceptmod/textsliders/pg09_audio.py", "conceptmod/textsliders/pg09_backends.py",
    ))
    core_root = core_source or os.environ.get("YUE2_CORE_SOURCE")
    core_files = (
        "packages/particle-sliders-core/src/particle_sliders/__init__.py",
        "packages/particle-sliders-core/src/particle_sliders/pg09.py",
        "packages/particle-sliders-core/src/particle_sliders/pg09_release_sources.json",
        "packages/particle-sliders-core/src/particle_sliders/pg09_continuous_profile.json",
        "packages/particle-sliders-core/src/particle_sliders/formulation.py",
        "packages/particle-sliders-core/src/particle_sliders/reference.py",
    )
    if core_root:
        package_root = Path(core_root).resolve() / "packages/particle-sliders-core/src/particle_sliders"
        core_files = tuple(str(p.relative_to(Path(core_root).resolve())) for p in sorted(package_root.rglob("*"))
                           if p.is_file() and p.suffix in (".py", ".json"))
    core, core_id = _source(core_root, CORE_PIN, core_files)
    core_src = core / "packages/particle-sliders-core/src"
    expected = {"particle_sliders": core_src / "particle_sliders/__init__.py",
                "particle_sliders.pg09": core_src / "particle_sliders/pg09.py",
                "conceptmod": audio / "conceptmod/__init__.py",
                "conceptmod.textsliders": audio / "conceptmod/textsliders/__init__.py",
                "conceptmod.textsliders.pg09_audio": audio / "conceptmod/textsliders/pg09_audio.py",
                "conceptmod.textsliders.pg09_backends": audio / "conceptmod/textsliders/pg09_backends.py"}
    for name, path in expected.items():
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() != path:
            raise ValueError(f"already imported {name} from an unqualified source: {module.__file__}")
    for path in (str(audio), str(core_src)):
        if path not in sys.path:
            sys.path.insert(0, path)
    for name, path in expected.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != path:
            raise ValueError(f"import origin differs from pinned integration source: {name}")
    return dict(audio=audio_id, core=core_id)
