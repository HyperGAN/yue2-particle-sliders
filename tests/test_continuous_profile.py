"""Explicit continuous source is a separate numerical cohort, never a release alias."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference
from yue2_sliders import response_runtime as runtime
from yue2_sliders.native_assets import verify_native, MODEL_REVISION

PROFILE = 'release-0.9.5'
PAIR = dict(pg_source_profile=PROFILE, penalty_input_domain='continuous')


def candidate_installed():
    from particle_sliders.pg09 import _released_sources
    try:
        _released_sources(PROFILE)
    except RuntimeError:
        return False
    return True


def config():
    return SessionConfig('yue2', 'continuous-source-cpu-fixture-only', dummy=True,
        rows=[dict(neutral='soft piano', positive='bright piano', lyrics='A new day begins'),
              dict(neutral='quiet flute', positive='strong flute', lyrics='The river carries light')],
        steps=2, seed=7, batch=2, history_tokens=4, ema_decay=.9, **PAIR)


def test_native_runtime_versions_keep_the_original_asset_lock():
    from yue2_sliders.native_assets import LOCK, VERSIONS
    assert VERSIONS == json.loads(LOCK.read_text())["runtime_versions"]


def test_descriptor_preserves_explicit_native_width_and_profile():
    native = runtime.capabilities(PAIR)
    assert native.response_shape == (1, 2048)
    assert native.pg_source_profile == PROFILE
    assert native.penalty_input_domain == 'continuous'
    assert runtime.capabilities({}).pg_source_profile == 'release-0.9.5'
    assert runtime.capabilities({}).penalty_input_domain == 'auto'
    for invalid in (dict(pg_source_profile='source-801ae1e-continuous-v1', penalty_input_domain='continuous'),
                    dict(pg_source_profile='release-0.9.4'),
                    dict(pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')):
        with pytest.raises(ValueError, match='explicitly qualified'):
            runtime.validate_native_options(invalid)
    assert not torch.cuda.is_initialized()


def test_native_source_candidate_requires_explicit_precision_and_revision_before_load():
    with pytest.raises(ValueError, match='explicit CUDA'):
        verify_native(PAIR)
    with pytest.raises(ValueError, match='exact full model revision'):
        verify_native(dict(PAIR, device='cuda:0', base_precision='bfloat16'))
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(not candidate_installed(), reason='requires exact published PG0.9.5 source environment')
@pytest.mark.parametrize('selection', ['raw', 'ema'])
def test_explicit_profile_cpu_fixture_exact_resume_and_native_only_exports(tmp_path, selection):
    """Real shared optimizers on a bounded random fixture, no pretrained claim."""
    cfg = config()
    with open_response(cfg) as full:
        first = full.step(); one = full.checkpoint()
        second = full.step(); expected = full.checkpoint()
        full.restore(one)
        assert full.step() == second
        assert full.checkpoint()['sha256'] == expected['sha256']
        native = full.game.backend
        with full._observation(11):
            if selection == 'ema':
                native.network.load_state_dict(full.ema.values)
            expected_response = native.response(token_ids=[12, 21, 30], strength=.5, seed=11)
        artifact = tmp_path / (selection + '.pt')
        packet = full.export_inference(artifact, selection=selection)
        assert packet['specification']['configuration']['pg_source_profile'] == PROFILE
        assert packet['specification']['configuration']['penalty_input_domain'] == 'continuous'
        assert full.game.state_dict()['config']['pg_source_profile'] == PROFILE
    with open_inference(artifact, operation='response', options={'device':'cpu'}) as exported:
        assert torch.equal(exported.response(token_ids=[12,21,30], strength=.5, seed=11), expected_response)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('tamper', ['none', 'value', 'extra-key', 'dtype'])
def test_loaded_checkpoint_checks_all_values_before_attachment(tmp_path, monkeypatch, tamper):
    hub = pytest.importorskip('huggingface_hub')
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from yue2_sliders.native_assets import verify_loaded_model, ARCHITECTURE
    checkpoint = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    save_file({'weight': checkpoint}, tmp_path/'model.safetensors')
    class SmallFrozenHost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config=SimpleNamespace(**ARCHITECTURE)
            self.register_buffer('weight', checkpoint.clone())
    host=SmallFrozenHost()
    monkeypatch.setattr(hub, 'snapshot_download', lambda *a, **k: str(tmp_path))
    if tamper=='value':host.weight[0,0]=9
    elif tamper=='extra-key':host.register_buffer('extra', torch.zeros(1))
    elif tamper=='dtype':host.weight=host.weight.float()
    if tamper!='none':
        with pytest.raises(ValueError, match='checkpoint (key inventory|tensor|device/precision) differs'):
            verify_loaded_model(host, {'device':'cpu'})
    else:
        proof=verify_loaded_model(host, {'device':'cpu'})
        assert proof['all_values_exact'] and proof['complete_checkpoint_tensors']==1
        assert proof['frozen_base_sha256']==runtime.frozen_digest(host)
    assert not torch.cuda.is_initialized()
