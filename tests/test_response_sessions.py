"""Installed-product response/inference conformance; authored, not yet executed."""
from copy import deepcopy
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation, open_session
from particle_sliders.response import open_response


def config():
    return SessionConfig('yue2', 'ownership-proposal-test-cohort', dummy=True,
        rows=[dict(neutral='a soft piano', positive='a loud guitar', lyrics='[verse]\nA train rolls'),
              dict(neutral='a quiet flute', positive='a bright brass', lyrics='[verse]\nThe river runs')],
        steps=4, seed=7, batch=2, sample_seeds=1, history_tokens=4, ema_decay=.9)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_response_split_resume_ema_and_separate_image_packet_contract(tmp_path):
    cfg = config()
    with open_response(cfg) as full:
        full.train()
        expected = full.checkpoint()
    with open_response(cfg) as first:
        first.train(stop_after=2)
        first.checkpoint(tmp_path / 'training.pt')
    with open_response(cfg, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        actual = resumed.checkpoint()
        assert_same(actual, expected)
        assert actual['schema'] == 'particle-sliders-response-session-v1-training'
        assert resumed.capabilities.modality == 'audio_ar_hidden'
        assert resumed.capabilities.response_shape == (1, 32)
        for selection in ('raw', 'ema'):
            packet = resumed.export(selection=selection)
            assert packet['schema'] == 'particle-sliders-response-session-v1-inference'
        before = deepcopy(actual)
        bad = deepcopy(actual)
        bad['sha256'] = '0' * 64
        with pytest.raises(CheckpointError):
            resumed.restore(bad)
        assert_same(resumed.checkpoint(), before)
        with pytest.raises(UnsupportedOperation):
            resumed.evaluate(strength=-.5)
        with pytest.raises(UnsupportedOperation):
            resumed.render()
    with pytest.raises(CheckpointError):
        open_session(cfg, resume=tmp_path / 'training.pt')


def test_native_width_and_unrecovered_profile_refuse_before_loading(monkeypatch):
    from yue2_sliders import backend
    def forbidden(*args, **kwargs):
        raise AssertionError('pretrained constructor reached')
    monkeypatch.setattr(backend, 'YuE2Backend', forbidden)
    options = config().options
    options['dummy'] = False
    with pytest.raises(ValueError, match='width >=170'):
        open_response(SessionConfig('yue2', 'native-refusal-only', **options))
    options.update(pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    with pytest.raises(ValueError, match='unrecovered'):
        open_response(SessionConfig('yue2', 'unrecovered-refusal-only', **options))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('selection', ['raw', 'ema'])
def test_native_only_fresh_process_without_training_imports(tmp_path, selection):
    cfg = config()
    artifact = tmp_path / (selection + '-v2.pt')
    with open_response(cfg) as run:
        run.train(stop_after=2)
        packet = run.export_inference(artifact, selection=selection)
        assert packet['selection'] == selection
        assert packet['schema'] == 'particle-sliders-native-inference-v2'
        assert all(row['neutral'] not in json.dumps(packet['specification']) for row in cfg.options['rows'])
        with run._observation(11):
            if selection == 'ema':
                run.game.network.load_state_dict(run.ema.values)
            expected = run.game.backend.response(token_ids=[12, 21, 30], strength=.5, seed=11)
    roots = list(dict.fromkeys(str(Path(metadata.distribution(name).locate_file('')).resolve())
        for name in ('yue2-particle-sliders', 'particle-sliders-core', 'particlegan')))
    output = tmp_path / (selection + '-response.pt')
    code = r'''
import json, sys, importlib.abc
sys.path[:0] = json.loads(sys.argv[1])  # Exact metadata-owned installed targets only.
class ForbidTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'yue2_sliders.lifecycle', 'yue2_sliders.train', 'yue2_sliders.response_provider', 'yue2_sliders.infer'}:
            raise AssertionError('native-only inference imported training: ' + fullname)
sys.meta_path.insert(0, ForbidTraining())
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import particle_sliders.pg09 as pg
import particle_sliders.pg09_audio as audio
from particle_sliders.inference import open_inference

def forbidden(*args, **kwargs):
    raise AssertionError('native-only inference constructed training components')
pg.ConditionalPG09.__init__ = forbidden
audio.AudioResponseCritic.__init__ = forbidden
torch.optim.Optimizer.__init__ = forbidden
with open_inference(sys.argv[2], operation='response', options={'device':'cpu'}) as run:
    result = run.response(token_ids=[12,21,30], strength=.5, seed=11)
    torch.save(result, sys.argv[3])
    try:
        run.train()
    except RuntimeError:
        pass
    else:
        raise AssertionError('native inference admitted training')
    print(json.dumps({'selection': run.selection, 'cuda': torch.cuda.is_initialized()}))
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONPATH', None)
    process = subprocess.run([sys.executable, '-I', '-c', code, json.dumps(roots), str(artifact), str(output)],
                             cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stdout + process.stderr
    receipt = json.loads(process.stdout)
    assert receipt == dict(selection=selection, cuda=False)
    assert torch.equal(torch.load(output, weights_only=True, map_location='cpu'), expected)
    with open_inference(artifact, operation='response', options={'device': 'cpu'}) as native:
        with pytest.raises(UnsupportedOperation):
            native.response(token_ids=[12], strength=-.5)
        with pytest.raises(ValueError, match='in-vocabulary'):
            native.response(token_ids=[True], strength=.5)
        with pytest.raises(ValueError, match='context limit'):
            native.response(token_ids=[12] * 257)


def test_binding_cleanup_retry_remains_available_after_session_close_conflict():
    from particle_sliders.pg09_backends import checkpoint_with_adapter_scale
    run = open_response(config())
    native = run.game.backend
    block = native.model.model.layers[0]
    # Exercise the same inherited ledger on a tiny registered native host block;
    # production width2048 remains refused and no native model is claimed here.
    handle = checkpoint_with_adapter_scale(block, native.network)
    native._checkpointing.append(handle)
    owned = block.forward
    foreign = lambda *args, **kwargs: None
    block.forward = foreign
    try:
        with pytest.raises(RuntimeError, match='ownership changed'):
            run.close()
        assert run.closed  # Current common policy; binding callback is the retry seam.
        assert block.forward is foreign
    finally:
        block.forward = owned
        run.binding.close()
    run.binding.close()
    assert 'forward' not in vars(block)
    assert not getattr(block, '_pg09_scale_checkpoint', False)
    assert all('forward' not in vars(module) for module in native.backend.model.modules()
               if isinstance(module, torch.nn.Linear))
