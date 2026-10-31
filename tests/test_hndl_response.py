"""Real HyperGAN develop/HNDL0.10/PG0.9.4 CPU fixtures; no pretrained claim."""
from copy import deepcopy
from importlib import metadata

import pytest
import torch

try:
    if metadata.version('particle-sliders-core') != '0.12.0' or metadata.version('hndl') != '0.10.0':
        pytest.skip('requires the separate core0.12/HNDL0.10 preview environment', allow_module_level=True)
except metadata.PackageNotFoundError:
    pytest.skip('requires the separate core0.12/HNDL0.10 preview environment', allow_module_level=True)

from particle_sliders.lifecycle import CheckpointError, SessionConfig
from particle_sliders.inference import open_inference
from particle_sliders.response import open_response
from yue2_sliders import hndl_response as product
from yue2_sliders.backend import YuE2Backend, attention_targets
from yue2_sliders.response_native import NativeOwner


def config():
    return SessionConfig('yue2-hndl', 'yue2-hndl-cpu-v1', dummy=True,
        rows=[dict(neutral='a soft piano', positive='a loud guitar', lyrics='[verse]\nA train rolls'),
              dict(neutral='a quiet flute', positive='a bright brass', lyrics='[verse]\nThe river runs')],
        steps=4, seed=7, batch=2, sample_seeds=1, history_tokens=4, ema_decay=.9)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def affine(dst, src):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)


def test_matched_product_network_outputs_bank_gradients_and_zero_strength():
    from yue2_sliders.slider import ParticleSlider
    from yue2_sliders.stamp import locked_stamp
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    torch.manual_seed(7)
    original = YuE2Backend(dummy=True)
    runtime = YuE2Backend(dummy=True)
    runtime.model.load_state_dict(original.model.state_dict())
    baseline = deepcopy(runtime.model.state_dict())
    owner = NativeOwner(runtime.model, slider_factory=product.HNDLSlider, projection_type=HNDLParticleProjection)
    old = ParticleSlider(original.model, locked_stamp()[0])
    try:
        new = owner.attach_slider(runtime.model)
        with torch.no_grad():
            new.particles.copy_(old.particles)
        for before, after in zip(old.adapters.values(), new.adapters.values()):
            torch.nn.init.normal_(before.lora_up.weight, std=.02)
            affine(after.graph['down'], before.lora_down)
            affine(after.graph['up'], before.lora_up)
            for family, layers in [('router', before.bridge.router), ('bridge', before.bridge.net)]:
                for index in (0, 2, 4, 6):
                    affine(after.graph[family + str(index)], layers[index])
        ids = [12, 21, 30]
        for strength in (0., .5, 1.):
            with old.scaled(strength), new.scaled(strength):
                expected, actual = original.hidden(ids), runtime.hidden(ids)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            if strength == 0:
                assert torch.equal(actual, expected)
        with old.scaled(.5), new.scaled(.5):
            expected, actual = original.hidden(ids)[:, -1, :3].sum(), runtime.hidden(ids)[:, -1, :3].sum()
        expected.backward()
        actual.backward()
        torch.testing.assert_close(new.particles.grad, old.particles.grad, atol=3e-6, rtol=3e-5)
        assert new.particles.grad.abs().sum() > 0
        assert all(not p.requires_grad and p.grad is None for p in runtime.model.parameters())
        assert all(torch.equal(value, baseline[key]) for key, value in runtime.model.state_dict().items())
        assert sum(p is new.particles for p in new.parameters()) == 1
    finally:
        owner.close()
    assert all('forward' not in vars(m) for m in attention_targets(runtime.model).values())


def test_real_stack_split_resume_ema_inference_and_wrong_source_rejection(tmp_path):
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    cfg = config()
    rng = torch.get_rng_state().clone()
    with open_response(cfg) as run:
        assert isinstance(run.game.critic, product.HNDLAudioCritic)
        assert all(isinstance(a, HNDLParticleProjection) for a in run.game.backend.slider.adapters.values())
        before = run.game.backend.slider.particles.detach().clone()
        run.train()
        expected = run.checkpoint()
        assert not torch.equal(before, run.game.backend.slider.particles)
    with open_response(cfg) as run:
        run.train(stop_after=2)
        packet = run.checkpoint(tmp_path / 'training.pt')
    with open_response(cfg, resume=tmp_path / 'training.pt') as run:
        run.train()
        same(run.checkpoint(), expected)
        for selection in ('raw', 'ema'):
            artifact = tmp_path / (selection + '.pt')
            run.export_inference(artifact, selection=selection)
            with open_inference(artifact, operation='response', options={'device': 'cpu'}) as native:
                assert torch.isfinite(native.response(token_ids=[12, 21, 30], strength=.5)).all()
        before = run.checkpoint()
        bad = deepcopy(packet)
        bad['manifest']['cohort'] = 'old-cohort'
        from particle_sliders.lifecycle import _hash
        bad['sha256'] = _hash({k: v for k, v in bad.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            run.restore(bad)
        same(run.checkpoint(), before)
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_native_width_refuses_before_loader_and_no_profile_override(monkeypatch):
    from yue2_sliders import backend
    def forbidden(*args, **kwargs):
        raise AssertionError('native pretrained loader reached')
    monkeypatch.setattr(backend, 'YuE2Backend', forbidden)
    options = config().options
    options['dummy'] = False
    with pytest.raises(ValueError, match='width >=170'):
        open_response(SessionConfig('yue2-hndl', 'native-refusal', **options))
    options.update(pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    with pytest.raises(ValueError, match='no domain override'):
        open_response(SessionConfig('yue2-hndl', 'wrong-profile', **options))


def test_builder_tamper_rejected_before_hooks_or_flags(monkeypatch):
    import particle_sliders.fibo_hndl as shared
    runtime = YuE2Backend(dummy=True)
    forwards = [m.forward for m in attention_targets(runtime.model).values()]
    flags = [p.requires_grad for p in runtime.model.parameters()]
    original = shared._sha
    monkeypatch.setattr(shared, '_sha', lambda p:
        '0' * 64 if str(p).endswith('hndl_networks.py') else original(p))
    with pytest.raises(ValueError, match='builder closure'):
        product.HNDLSlider(runtime.model)
    assert forwards == [m.forward for m in attention_targets(runtime.model).values()]
    assert flags == [p.requires_grad for p in runtime.model.parameters()]


def test_hndl_audio_readout_matches_original_scores_and_penalty_derivatives():
    from particle_sliders.pg09_audio import AudioResponseCritic
    old, new = AudioResponseCritic(32, 32), product.HNDLAudioCritic(32, 32)
    affine(new.graph['hidden'], old.net[0])
    affine(new.graph['logits'], old.net[2])
    response = torch.randn(2, 32, requires_grad=True)
    twin = response.detach().clone().requires_grad_()
    condition = torch.randn(2, 32)
    left, right = old(response, condition), new(twin, condition)
    torch.testing.assert_close(right, left)
    g0 = torch.autograd.grad(left.sum(), response, create_graph=True)[0]
    g1 = torch.autograd.grad(right.sum(), twin, create_graph=True)[0]
    torch.testing.assert_close(g1, g0)
    d0 = torch.autograd.grad(g0.square().sum(), old.net[0].weight)[0]
    d1 = torch.autograd.grad(g1.square().sum(), new.graph['hidden'].weight)[0]
    torch.testing.assert_close(d1, d0)


@pytest.mark.parametrize('strength', [.5, 1.])
def test_hndl_deferred_backward_checkpoint_scale_and_ownership(strength):
    from particle_sliders.pg09_backends import checkpoint_with_adapter_scale
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    def gradients(checkpointed):
        torch.manual_seed(7)
        runtime = YuE2Backend(dummy=True)
        owner = NativeOwner(runtime.model, slider_factory=product.HNDLSlider, projection_type=HNDLParticleProjection)
        slider = owner.attach_slider(runtime.model)
        with torch.no_grad():
            for adapter in slider.adapters.values():
                adapter.lora_up.weight.fill_(.02)
        native = product.NativeResponse(runtime, slider, model_identity={'fixture': 'tiny'},
            pg_source_profile=product.PROFILE)
        owner.adapter = native
        if checkpointed:
            for layer in runtime.model.model.layers:
                native._checkpointing.append(checkpoint_with_adapter_scale(layer.self_attn.q_proj, native.network))
        with native.network.scaled(strength):
            loss = runtime.hidden([12, 21, 30])[:, -1, :3].sum()
        owner.close()
        loss.backward()
        assert all(p.grad is None for p in runtime.model.parameters())
        assert all('forward' not in vars(m) for m in attention_targets(runtime.model).values())
        return [None if p.grad is None else p.grad.clone() for p in slider.parameters()]
    actual, expected = gradients(True), gradients(False)
    same(actual, expected)
    assert any(p is not None and p.abs().sum() for p in actual)


def test_preview_metadata_and_native_only_reload_without_training_construction(tmp_path, monkeypatch):
    from importlib import metadata
    import os
    import subprocess
    import sys
    distribution = metadata.distribution('yue2-particle-sliders')
    assert distribution.version == '0.3.0.dev0'
    from packaging.specifiers import SpecifierSet
    assert SpecifierSet(distribution.metadata['Requires-Python']) == SpecifierSet('>=3.11,<3.15')
    core = next(r for r in distribution.requires if r.startswith('particle-sliders-core'))
    assert '[hypergan-hndl]' in core and '@' + product.CORE_PIN in core
    assert {(ep.group, ep.name) for ep in distribution.entry_points} == {
        ('particle_sliders.response_backends', 'yue2-hndl'),
        ('particle_sliders.inference_backends', 'yue2-hndl')}
    with open_response(config()) as run:
        run.train(stop_after=1)
        run.export_inference(tmp_path / 'native.pt')
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'yue2_sliders.lifecycle', 'yue2_sliders.train', 'yue2_sliders.response_provider'}:
            raise AssertionError('training import: ' + fullname)
sys.meta_path.insert(0, Guard())
from particle_sliders.inference import open_inference
from particle_sliders import ConditionalPG09
from yue2_sliders.hndl_response import HNDLAudioCritic
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('training construction')
ConditionalPG09.__init__ = forbidden
HNDLAudioCritic.__init__ = forbidden
torch.optim.Optimizer.__init__ = forbidden
with open_inference(sys.argv[1], operation='response', options={'device': 'cpu'}) as run:
    value = run.response(token_ids=[12,21,30], strength=.5)
    assert tuple(value.shape) == (1,32) and torch.isfinite(value).all()
assert not torch.cuda.is_initialized()
print('native-only HNDL inference passed')
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(tmp_path / 'native.pt')],
        cwd=tmp_path, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1'),
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def native_fixture(seed=7):
    """Actual pinned native YuE2 classes with random tiny weights; no Hub load."""
    pytest.importorskip('yue2.modeling_yue2', reason='requires the separate pinned native API fixture slot')
    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
    from yue2_sliders.defaults import UPSTREAM_REVISION
    from pathlib import Path
    import hashlib
    import inspect
    direct = __import__('json').loads(metadata.distribution('yue2-infer').read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id'] == UPSTREAM_REVISION
    torch.manual_seed(seed)
    config = YuE2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, intermediate_size=64, vocab_size=256,
        max_position_embeddings=256, latent_dim=8, max_latent_frames=32)
    runtime = object.__new__(YuE2Backend)
    runtime.model = YuE2ForCausalLM(config).eval().requires_grad_(False)
    runtime.dummy = False
    runtime.identity = dict(fixture='random tiny native YuE2ForCausalLM; not pretrained',
        config=config.to_dict(), upstream_revision=UPSTREAM_REVISION,
        native_modeling_sha256=hashlib.sha256(Path(inspect.getsourcefile(YuE2ForCausalLM)).read_bytes()).hexdigest())
    assert runtime.identity['native_modeling_sha256'] == '62d2bd758c8fff1a0d64584a699343076fca9a7880db5d8e19263bc30ae79d7c'
    return runtime


def test_genuine_native_yue2_grouped_query_response_and_bank_vjp_parity():
    from yue2_sliders.slider import ParticleSlider
    from yue2_sliders.stamp import locked_stamp
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    original, runtime = native_fixture(), native_fixture()
    before = deepcopy(runtime.model.state_dict())
    old = ParticleSlider(original.model, locked_stamp()[0])
    owner = NativeOwner(runtime.model, slider_factory=product.HNDLSlider, projection_type=HNDLParticleProjection)
    try:
        new = owner.attach_slider(runtime.model)
        with torch.no_grad():
            new.particles.copy_(old.particles)
        for prior, current in zip(old.adapters.values(), new.adapters.values()):
            torch.nn.init.normal_(prior.lora_up.weight, std=.02)
            affine(current.graph['down'], prior.lora_down)
            affine(current.graph['up'], prior.lora_up)
            for family, layers in [('router', prior.bridge.router), ('bridge', prior.bridge.net)]:
                for index in (0, 2, 4, 6):
                    affine(current.graph[family + str(index)], layers[index])
        assert runtime.model.model.layers[0].self_attn.k_proj.out_features == 16
        for strength in (0., .5, 1.):
            with old.scaled(strength), new.scaled(strength):
                expected, actual = original.hidden([12, 21, 30]), runtime.hidden([12, 21, 30])
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            if strength == 0:
                assert torch.equal(actual, expected)
        with old.scaled(.5), new.scaled(.5):
            original.hidden([12, 21, 30])[:, -1, :3].sum().backward()
            runtime.hidden([12, 21, 30])[:, -1, :3].sum().backward()
        torch.testing.assert_close(new.particles.grad, old.particles.grad, atol=3e-6, rtol=3e-5)
        assert new.particles.grad.abs().sum() > 0
        assert all(p.grad is None and not p.requires_grad for p in runtime.model.parameters())
        assert all(torch.equal(value, before[key]) for key, value in runtime.model.state_dict().items())
    finally:
        owner.close()
    assert all('forward' not in vars(m) for m in attention_targets(runtime.model).values())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_genuine_native_yue2_shared_game_updates_and_exact_restore(preset):
    from particle_sliders.pg094 import ConditionalPG094
    from particle_sliders.pg09_audio import AudioRow, audio_conditions
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    def build():
        runtime = native_fixture()
        owner = NativeOwner(runtime.model, slider_factory=product.HNDLSlider, projection_type=HNDLParticleProjection)
        try:
            slider = owner.attach_slider(runtime.model)
            native = product.NativeResponse(runtime, slider, model_identity=runtime.identity,
                checkpoint_modules=list(runtime.model.model.layers), pg_source_profile=product.PROFILE)
            owner.adapter = native
            rows = [native.prepare(AudioRow((12,21,30), (13,22,30))),
                    native.prepare(AudioRow((14,23,31), (15,24,31)))]
            critic = product.HNDLAudioCritic(32, audio_conditions(rows).shape[1])
            game = ConditionalPG094(native, rows, critic, particles=native.particles,
                conditions=audio_conditions(rows), response_domain='continuous_vector',
                preset=preset, batch_size=2, seed=7, total_steps=2, pg_source_profile=product.PROFILE)
            return game, owner
        except BaseException:
            owner.close()
            raise
    game, owner = build()
    before = game.backend.slider.particles.detach().clone()
    critic_before = deepcopy(game.critic.state_dict())
    try:
        game.step()
        packet = game.state_dict()
        game.step()
        expected = game.state_dict()
        assert not torch.equal(before, game.backend.slider.particles)
        assert any(not torch.equal(value, critic_before[key]) for key, value in game.critic.state_dict().items())
        assert all(p.grad is None for p in game.backend.model.parameters())
    finally:
        owner.close()
    resumed, owner = build()
    try:
        resumed.load_state_dict(packet)
        resumed.step()
        same(resumed.state_dict(), expected)
    finally:
        owner.close()
