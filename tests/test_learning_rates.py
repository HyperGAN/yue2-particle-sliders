"""LR controls bind actual shared optimizers and exact continuation."""
import pytest
from yue2_sliders.learning_rates import recipe_overrides


@pytest.mark.parametrize('key', ['lr', 'd_lr_mult', 'prior_lr_mult'])
@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), '0.001'])
def test_invalid_rates(key, value):
    with pytest.raises(ValueError):
        recipe_overrides(**{key: value})


def test_optional_rates_keep_preset_defaults():
    assert recipe_overrides() == {}


def configuration(tmp_path, **rates):
    from particle_sliders.lifecycle import SessionConfig
    return SessionConfig('yue2','lr-controls-v1',dummy=True,steps=4,batch=2,seed=7,sample_seeds=1,history_tokens=4,rows=[dict(neutral='a soft piano',positive='a loud guitar',lyrics='[verse] A train rolls'),dict(neutral='a quiet flute',positive='a bright brass',lyrics='[verse] The river runs')],**rates)


def test_actual_rates_and_exact_resume(tmp_path):
    import torch
    from particle_sliders.response import open_response
    from particle_sliders.lifecycle import CheckpointError, SessionConfig
    torch.set_num_threads(1)
    config=configuration(tmp_path,lr=2e-5,d_lr_mult=3,prior_lr_mult=4)
    with open_response(config) as full:
        game=full.game
        assert (game.recipe.lr,game.recipe.d_lr_mult,game.recipe.prior_lr_mult)==(2e-5,3,4)
        for group in game.opt_g.param_groups:
            for parameter in group['params']:
                assert group['lr']==pytest.approx(8e-5 if parameter is game.prior.z else 2e-5)
        assert all(group['lr']==pytest.approx(6e-5) for group in game.opt_d.param_groups)
        full.train()
        expected=full.checkpoint()['sha256']
        full.export_inference(tmp_path/'native.pt')
    with open_response(config) as partial:
        partial.train(stop_after=2)
        packet=partial.checkpoint()
    with open_response(config) as resumed:
        resumed.restore(packet)
        resumed.train()
        assert resumed.checkpoint()['sha256']==expected
    changed=SessionConfig(config.backend,config.cohort,**{**config.options,'lr':3e-5})
    with open_response(changed) as other:
        with pytest.raises(CheckpointError):
            other.restore(packet)
    assert not torch.cuda.is_initialized()


def test_legacy_hndl_archive(tmp_path):
    import importlib.util
    import json
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('yue2_lr_archive',root/'release_tools/build_hndl_preview.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest=json.loads((root/'cohorts/pre-lr-hndl/manifest.json').read_text())
    for relative in manifest['files']:
        (tmp_path/relative).parent.mkdir(parents=True,exist_ok=True)
    module.restore_pre_lr_runtime(root,tmp_path)
    for relative,digest in manifest['files'].items():
        assert hashlib.sha256((tmp_path/relative).read_bytes()).hexdigest()==digest
