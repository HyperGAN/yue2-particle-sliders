"""Current official PG094 lifecycle identity and installed CPU conformance."""
from copy import deepcopy
import json
from importlib import metadata
from pathlib import Path
import shutil

import pytest
import torch

from particle_sliders.lifecycle import CheckpointError, SessionConfig
from particle_sliders.response import open_response
from yue2_sliders import response_runtime as runtime


def config(preset='bispf_optimized'):
    return SessionConfig('yue2', 'pg094-cpu-regression-' + preset, dummy=True,
        rows=[dict(neutral='soft piano', positive='loud guitar', lyrics='A train rolls'),
              dict(neutral='quiet flute', positive='bright brass', lyrics='The river runs')],
        steps=2, seed=7, batch=2, sample_seeds=1, history_tokens=4,
        preset=preset, ema_decay=.9)


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_canonical_lifecycle_trains_with_official_pg094_and_keeps_base_frozen(preset):
    with open_response(config(preset)) as run:
        assert run.capabilities.pg_version == '0.9.4'
        assert run.capabilities.pg_source_profile == 'release-0.9.4'
        before = run.game.backend.slider.particles.detach().clone()
        base = deepcopy(run.game.backend.model.state_dict())
        critic = deepcopy(run.game.critic.state_dict())
        run.train()
        assert not torch.equal(before, run.game.backend.slider.particles)
        assert any(not torch.equal(v, critic[k]) for k, v in run.game.critic.state_dict().items())
        assert all(torch.equal(v, base[k]) for k, v in run.game.backend.model.state_dict().items())
        assert all(not p.requires_grad and p.grad is None for p in run.game.backend.model.parameters())
        state = run.game.state_dict()
        assert state['config']['pg_version'] == '0.9.4'
        assert state['config']['pg_source_profile'] == 'release-0.9.4'
        state['config']['pg_version'] = '0.9.1'
        with pytest.raises(ValueError, match='version|cohort|signature|mismatch|profile'):
            run.game.load_state_dict(state)


@pytest.mark.parametrize('provenance', ['wrong-wheel-hash', 'missing-wheel-hash', 'missing', 'wrong-vcs', 'exact-vcs'])
def test_current_core_archive_or_exact_git_provenance_required(monkeypatch, provenance):
    original = runtime.metadata.distribution
    distribution = original('particle-sliders-core')
    records = {
        'wrong-wheel-hash': {'archive_info': {'hashes': {'sha256': '0' * 64}}},
        'missing-wheel-hash': {'archive_info': {}},
        'missing': {},
        'wrong-vcs': {'url': 'https://github.com/HyperGAN/particle-sliders.git',
                      'vcs_info': {'vcs': 'git', 'commit_id': '0' * 40}},
        'exact-vcs': {'url': 'https://github.com/HyperGAN/particle-sliders.git',
                      'vcs_info': {'vcs': 'git', 'commit_id': runtime.CORE_PIN}},
    }
    class CoreView:
        def __getattr__(self, name):
            return getattr(distribution, name)
        def read_text(self, name):
            return json.dumps(records[provenance]) if name == 'direct_url.json' else distribution.read_text(name)
    monkeypatch.setattr(runtime.metadata, 'distribution', lambda name:
        CoreView() if name == 'particle-sliders-core' else original(name))
    if provenance == 'exact-vcs':
        assert runtime.verify_runtime()['core_commit'] == runtime.CORE_PIN
    else:
        with pytest.raises(ValueError, match='wheel SHA256|archive or Git commit provenance'):
            runtime.verify_runtime()


@pytest.mark.parametrize('tamper', ['python', 'graph', 'nested-data'])
def test_full_current_source_closure_rejects_tampering_before_native_load(tmp_path, monkeypatch, tamper):
    from yue2_sliders import backend
    original = runtime.metadata.distribution
    distribution = original('particle-sliders-core')
    for name in ('particle_sliders', 'concept_slider_core'):
        shutil.copytree(distribution.locate_file(name), tmp_path / name)
    source = tmp_path / {'python': 'particle_sliders/pg09.py',
        'graph': 'particle_sliders/fibo_projection.hndl',
        'nested-data': 'particle_sliders/new-data/extra.bin'}[tamper]
    source.parent.mkdir(exist_ok=True)
    source.write_bytes((source.read_bytes() if source.exists() else b'') + b'\n# tamper\n')
    class RelocatedCore:
        def __getattr__(self, name):
            return getattr(distribution, name)
        def locate_file(self, path):
            return tmp_path / path
    monkeypatch.setattr(runtime.metadata, 'distribution', lambda name:
        RelocatedCore() if name == 'particle-sliders-core' else original(name))
    def forbidden(*args, **kwargs):
        raise AssertionError('native loader reached before verification')
    monkeypatch.setattr(backend, 'YuE2Backend', forbidden)
    with pytest.raises(ValueError, match='shared source closure'):
        open_response(config())
    assert not torch.cuda.is_initialized()


def test_archived_pg091_lifecycle_lock_is_distinct_from_current():
    from yue2_sliders.resources import CONFIGS
    archived = json.loads((CONFIGS / 'lifecycle-pg091.lock.json').read_text())
    current = json.loads(runtime.LOCK.read_text())
    assert archived['core']['commit'] == 'f6807a91546e43e198645a3f8925d2dc7c9e03b4'
    assert archived['particlegan']['version'] == '0.9.1'
    assert current['core']['commit'] == 'ddb41423bcefc1fe76cd94d0f9b367353b3ea137'
    assert current['particlegan']['version'] == '0.9.4'
    assert current['particlegan']['release_commit'] == '330eb94ee748cdcad131b0275d8de75fd5a7f96f'


def test_unified_execution_uses_the_installed_response_backend_in_its_runtime(tmp_path):
    """Keep the newly merged shared execution API in the canonical source cohort."""
    import sys
    from particle_sliders.execution import resolve_request, execute_resolved_request
    profile = dict(schema='particle-slider-profile-v1', name='yue2-pg094-cpu',
        product='yue2-particle-sliders', config_kind='session', interface='response',
        backend='yue2', status='fixture', runtime={'python': sys.executable},
        capabilities={'resume': True, 'periodic_checkpoint': True, 'check': True})
    request = resolve_request(config().as_dict(), base=tmp_path, profile=profile,
        output=tmp_path / 'run', checkpoint_every=1, no_reporting=True)
    result = execute_resolved_request(request)
    assert result['status'] == 'completed'
    assert result['completed_steps'] == 2
    assert Path(result['checkpoint']).is_dir()
    assert (Path(result['checkpoint']) / 'training.pt').is_file()
    assert Path(result['adapter']).is_file()
    assert len(list((tmp_path / 'run' / 'artifacts' / 'checkpoints').iterdir())) == 2
    assert not torch.cuda.is_initialized()
