"""Run outside the checkout against an installed product wheel; CPU only."""
import hashlib
from importlib import metadata
import json
from pathlib import Path

import pytest

import yue2_sliders
_distribution = metadata.distribution('yue2-particle-sliders')
if Path(yue2_sliders.__file__).resolve().parent != Path(_distribution.locate_file('yue2_sliders')).resolve():
    pytest.skip('run outside the checkout against the installed product wheel', allow_module_level=True)


def test_installed_data_and_entrypoints_keep_their_cohorts():
    from yue2_sliders.resources import CONFIGS
    from yue2_sliders.integration import CORE_PIN as LEGACY_PIN, LOCK_PATH
    from yue2_sliders.response_runtime import CORE_PIN, LOCK, verify_runtime
    from yue2_sliders.samples import catalog_controls, featured_links, load_sample_cards

    distribution = metadata.distribution('yue2-particle-sliders')
    package = Path(distribution.locate_file('yue2_sliders')).resolve()
    assert CONFIGS == package / '_assets'  # No repository data fallback in a wheel.
    assert len(catalog_controls()) == 16
    assert len(featured_links(load_sample_cards())) == 3
    assert LOCK_PATH == CONFIGS / 'shared-runtime.lock.json'
    assert json.loads(LOCK_PATH.read_text())['core']['commit'] == LEGACY_PIN
    lock = json.loads(LOCK.read_text())
    assert lock['core']['commit'] == CORE_PIN
    assert CORE_PIN != LEGACY_PIN
    requirement = next(r for r in distribution.requires if r.startswith('particle-sliders-core'))
    assert '@' + CORE_PIN + '#subdirectory=packages/particle-sliders-core' in requirement
    assert verify_runtime()['core_commit'] == CORE_PIN
    registrations = {(ep.group, ep.name): ep for ep in distribution.entry_points}
    for group, module in [('response_backends', 'lifecycle'), ('inference_backends', 'response_native')]:
        ep = registrations[('particle_sliders.' + group, 'yue2')]
        assert ep.value.startswith('yue2_sliders.' + module + ':')
        assert ep.load().name == 'yue2'


def test_digest_supports_the_declared_python310_api(tmp_path, monkeypatch):
    from yue2_sliders.backend import file_digest
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    data = b'YuE2 provenance\x00' * 100000  # Cross the streaming chunk boundary.
    path = tmp_path / 'model-fragment'
    path.write_bytes(data)
    assert file_digest(path) == hashlib.sha256(data).hexdigest()


def test_new_shared_version_is_rejected_without_touching_the_lock(monkeypatch):
    from yue2_sliders import response_runtime as runtime
    lock = runtime.LOCK.read_bytes()
    original = runtime.metadata.distribution

    class NewCore:
        version = '0.12.0'
        def read_text(self, name):
            return '{}'

    monkeypatch.setattr(runtime.metadata, 'distribution', lambda name:
                        NewCore() if name == 'particle-sliders-core' else original(name))
    with pytest.raises(ValueError, match='locked wheel'):
        runtime.verify_runtime()
    assert runtime.LOCK.read_bytes() == lock
