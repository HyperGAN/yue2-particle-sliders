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


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_shipped_prepare_split_resume_ema_inference_and_wrong_cohort_rejection(tmp_path, preset):
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    from particle_sliders import ConditionalPG09
    cfg = SessionConfig('yue2-hndl', 'shipped-prepare-' + preset, **dict(config().options, preset=preset))
    rng = torch.get_rng_state().clone()
    with open_response(cfg) as run:
        assert type(run.game) is ConditionalPG09  # Registry invokes the shipped prepare(), not the direct alias fixture.
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
            with run._observation(11):
                if selection == 'ema':
                    run.game.network.load_state_dict(run.ema.values)
                expected_response = run.game.backend.response(token_ids=[12, 21, 30], strength=.5, seed=11)
                expected_off = run.game.backend.response(token_ids=[12, 21, 30], strength=0., seed=11)
            with open_inference(artifact, operation='response', options={'device': 'cpu'}) as native:
                assert torch.equal(native.response(token_ids=[12, 21, 30], strength=.5, seed=11), expected_response)
                assert torch.equal(native.response(token_ids=[12, 21, 30], strength=0., seed=11), expected_off)
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


def test_actual_changed_hndl_graph_source_rejects_checkpoint_without_mutation(tmp_path, monkeypatch):
    import shutil
    with open_response(config()) as original:
        original.train(stop_after=2)
        packet = original.checkpoint()
    for name in ('hndl_response.py', 'hndl-runtime.lock.json', 'hndl_audio_critic.hndl'):
        shutil.copyfile(product.ROOT / name, tmp_path / name)
    graph = tmp_path / 'hndl_audio_critic.hndl'
    graph.write_text(graph.read_text() + '\n# Different actual source; no cohort metadata edited.\n')
    monkeypatch.setattr(product, 'ROOT', tmp_path)
    with open_response(config()) as changed:
        before = changed.checkpoint()
        with pytest.raises(CheckpointError):
            changed.restore(packet)
        same(changed.checkpoint(), before)


@pytest.mark.parametrize('tamper', ['python', 'graph', 'nested-data'])
def test_actual_changed_shared_source_rejected_before_native_mutation(tmp_path, monkeypatch, tamper):
    import shutil
    distribution = metadata.distribution('particle-sliders-core')
    for name in ('particle_sliders', 'concept_slider_core'):
        shutil.copytree(distribution.locate_file(name), tmp_path / name)
    source = tmp_path / {'python': 'particle_sliders/pg09.py',
                         'graph': 'particle_sliders/fibo_projection.hndl',
                         'nested-data': 'particle_sliders/new-data/extra.bin'}[tamper]
    source.parent.mkdir(exist_ok=True)
    source.write_bytes((source.read_bytes() if source.exists() else b'') + b'\n# Actual copied source/data tamper.\n')
    original = product.metadata.distribution
    class RelocatedCore:
        def __getattr__(self, name):
            return getattr(distribution, name)
        def locate_file(self, path):
            return tmp_path / path
    monkeypatch.setattr(product.metadata, 'distribution', lambda name:
                        RelocatedCore() if name == 'particle-sliders-core' else original(name))
    runtime = YuE2Backend(dummy=True)
    before = deepcopy(runtime.model.state_dict())
    forwards = [module.forward for module in attention_targets(runtime.model).values()]
    flags = [p.requires_grad for p in runtime.model.parameters()]
    with pytest.raises(ValueError, match='shared source closure'):
        product.HNDLSlider(runtime.model)
    assert forwards == [module.forward for module in attention_targets(runtime.model).values()]
    assert flags == [p.requires_grad for p in runtime.model.parameters()]
    same(runtime.model.state_dict(), before)


@pytest.mark.parametrize('provenance', ['wrong-wheel-hash', 'missing-wheel-hash', 'missing', 'wrong-vcs', 'exact-vcs'])
def test_core_archive_or_vcs_provenance_is_enforced(monkeypatch, provenance):
    import json
    distribution = metadata.distribution('particle-sliders-core')
    records = {
        'wrong-wheel-hash': {'archive_info': {'hashes': {'sha256': '0' * 64}}},
        'missing-wheel-hash': {'archive_info': {}},
        'missing': {},
        'wrong-vcs': {'url': 'https://github.com/HyperGAN/particle-sliders.git',
                      'vcs_info': {'vcs': 'git', 'commit_id': '0' * 40}},
        'exact-vcs': {'url': 'https://github.com/HyperGAN/particle-sliders.git',
                      'vcs_info': {'vcs': 'git', 'commit_id': product.CORE_PIN}},
    }
    original = product.metadata.distribution
    class CoreView:
        def __getattr__(self, name):
            return getattr(distribution, name)
        def read_text(self, name):
            return json.dumps(records[provenance]) if name == 'direct_url.json' else distribution.read_text(name)
    monkeypatch.setattr(product.metadata, 'distribution', lambda name:
                        CoreView() if name == 'particle-sliders-core' else original(name))
    if provenance == 'exact-vcs':
        assert product.verify_runtime()[1]['core_commit'] == product.CORE_PIN
    else:
        with pytest.raises(ValueError, match='wheel SHA256|archive or Git commit provenance'):
            product.verify_runtime()


def test_float32_adapter_conversion_preserves_registered_frozen_bf16_host():
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    runtime = YuE2Backend(dummy=True)
    runtime.model.to(dtype=torch.bfloat16)
    before = deepcopy(runtime.model.state_dict())
    ids_before = {id(p) for p in runtime.model.parameters()}
    pointers_before = [p.data_ptr() for p in runtime.model.parameters()]
    with torch.no_grad():
        baseline = runtime.model.forward_hidden(torch.tensor([[12, 21, 30]])).clone()
    owner = NativeOwner(runtime.model, slider_factory=product.HNDLSlider, projection_type=HNDLParticleProjection)
    try:
        slider = owner.attach_slider(runtime.model)
        slider.to(dtype=torch.float32)
        assert not ids_before.intersection(id(p) for p in slider.parameters())
        assert all(p.dtype == torch.float32 for p in slider.parameters())
        assert all(p.dtype == torch.bfloat16 and not p.requires_grad and p.grad is None
                   for p in runtime.model.parameters())
        assert pointers_before == [p.data_ptr() for p in runtime.model.parameters()]
        same(runtime.model.state_dict(), before)
        with torch.no_grad(), slider.scaled(0):
            actual = runtime.model.forward_hidden(torch.tensor([[12, 21, 30]]))
        assert actual.dtype == torch.bfloat16 and torch.equal(actual, baseline)
    finally:
        owner.close()
    same(runtime.model.state_dict(), before)


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
    assert distribution.version == '0.3.0.dev1'
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


def test_modern_close_conflict_preserves_foreign_forward_and_retries():
    from particle_sliders.pg09_backends import checkpoint_with_adapter_scale
    run = open_response(config())
    native = run.game.backend
    block = native.model.model.layers[0].self_attn.q_proj
    native._checkpointing.append(checkpoint_with_adapter_scale(block, native.network))
    owned = block.forward
    foreign = lambda *args, **kwargs: None
    block.forward = foreign
    try:
        with pytest.raises(RuntimeError, match='ownership changed'):
            run.close()
        assert not run.closed
        assert block.forward is foreign
    finally:
        block.forward = owned
        run.close()
    assert run.closed
    run.close()
    assert all('forward' not in vars(module) for module in attention_targets(native.model).values())
    assert not any(getattr(module, '_pg09_scale_checkpoint', False) for module in native.model.modules())


def test_modern_partial_setup_failure_restores_staged_host_and_rng(monkeypatch):
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    runtime = YuE2Backend(dummy=True)
    runtime.model.train()
    before = deepcopy(runtime.model.state_dict())
    rng = torch.get_rng_state().clone()
    original = HNDLParticleProjection.__init__
    calls = 0
    def interrupted(self, *args, **kwargs):
        nonlocal calls
        original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError('partial modern setup fixture')
    monkeypatch.setattr(HNDLParticleProjection, '__init__', interrupted)
    with pytest.raises(RuntimeError, match='partial modern setup fixture'):
        open_response(config(), runtime=runtime)
    same(runtime.model.state_dict(), before)
    assert runtime.model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(not p.requires_grad and p.grad is None for p in runtime.model.parameters())
    assert all('forward' not in vars(module) for module in attention_targets(runtime.model).values())
