"""Native construction preserves callers and never seeds another visible GPU."""
from types import SimpleNamespace

import pytest
import torch

from yue2_sliders import response_native as native


def forbidden(*args, **kwargs):
    raise AssertionError('an all-device RNG operation was attempted')


@pytest.fixture
def cuda_generators(monkeypatch):
    # Three visible devices are simulated without initializing CUDA. Each uses
    # an independent real generator so fork_rng's save/restore path is exercised.
    generators = tuple(torch.Generator().manual_seed(100 + i) for i in range(3))
    visited = []
    monkeypatch.setattr(torch.cuda, 'default_generators', generators)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: (
        visited.append(('get', device)), generators[device].get_state())[1])
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state, device: (
        visited.append(('set', device)), generators[device].set_state(state))[1])
    monkeypatch.setattr(torch.cuda, 'device_count', forbidden)
    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    monkeypatch.setattr(torch.cuda, 'manual_seed', forbidden)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'seed_all', forbidden)
    return generators, visited


@pytest.mark.parametrize('fail', [False, True])
def test_only_requested_visible_device_is_seeded_and_every_state_is_restored(cuda_generators, fail):
    generators, visited = cuda_generators
    cpu_before = torch.get_rng_state().clone()
    before = [generator.get_state().clone() for generator in generators]
    expected = torch.rand(8, generator=torch.Generator().manual_seed(7))

    def construct():
        with native.native_initialization_rng({'device': 'cuda:1', 'base_seed': 7}):
            assert torch.equal(torch.rand(8), expected)
            assert torch.equal(torch.rand(8, generator=generators[1]), expected)
            assert torch.equal(generators[0].get_state(), before[0])
            assert torch.equal(generators[2].get_state(), before[2])
            if fail:
                raise RuntimeError('partial loader/attachment failure')

    if fail:
        with pytest.raises(RuntimeError, match='partial loader'):
            construct()
    else:
        construct()
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert all(torch.equal(generator.get_state(), state) for generator, state in zip(generators, before))
    assert visited == [('get', 1), ('set', 1)]
    assert not torch.cuda.is_initialized()


def test_cpu_native_scope_does_not_inspect_cuda(cuda_generators):
    _, visited = cuda_generators
    before = torch.get_rng_state().clone()
    with native.native_initialization_rng({'device': 'cpu', 'base_seed': 11}):
        first = torch.rand(5)
    with native.native_initialization_rng({'device': 'cpu', 'base_seed': 11}):
        assert torch.equal(first, torch.rand(5))
    assert torch.equal(before, torch.get_rng_state())
    assert visited == []


@pytest.mark.parametrize('device', ['cuda', 'meta', 'mps'])
def test_ambiguous_or_unsupported_device_refuses_before_seeding(cuda_generators, device):
    _, visited = cuda_generators
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='explicit CUDA device index'):
        with native.native_initialization_rng({'device': device}):
            pytest.fail('invalid device reached construction')
    assert torch.equal(before, torch.get_rng_state())
    assert visited == []


@pytest.mark.parametrize('fail', [False, True])
def test_public_native_builder_scopes_the_entire_constructor(monkeypatch, cuda_generators, fail):
    from yue2_sliders import native_assets
    generators, visited = cuda_generators
    monkeypatch.setattr(native, 'validate_native_options', lambda options: None)
    monkeypatch.setattr(native, 'verify_runtime', lambda options: {'verified': True})
    monkeypatch.setattr(native, 'capabilities', lambda options: SimpleNamespace(response_shape=(1, 2048)))
    monkeypatch.setattr(native, 'require_audio_width', lambda *args, **kwargs: None)
    monkeypatch.setattr(native, 'engine_options', lambda options: {})
    monkeypatch.setattr(native_assets, 'verify_native', lambda options: {'verified': True})
    cpu_before = torch.get_rng_state().clone()
    before = [generator.get_state().clone() for generator in generators]

    def construction(options, runtime, checkpoint_training, dependencies, assets):
        assert dependencies == assets == {'verified': True}
        result = torch.rand(4) + torch.rand(4, generator=generators[2])
        if fail:
            raise RuntimeError('constructor failed')
        return result

    monkeypatch.setattr(native, '_build_native', construction)
    if fail:
        with pytest.raises(RuntimeError, match='constructor failed'):
            native.build_native({'device': 'cuda:2'})
    else:
        assert torch.equal(native.build_native({'device': 'cuda:2'}), native.build_native({'device': 'cuda:2'}))
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert all(torch.equal(generator.get_state(), state) for generator, state in zip(generators, before))
    assert all(device == 2 for _, device in visited)


@pytest.mark.parametrize('fail', [False, True])
def test_training_prepare_scopes_provider_critic_game_and_failure(monkeypatch, cuda_generators, fail):
    from yue2_sliders import lifecycle, native_assets, response_runtime
    generators, visited = cuda_generators
    monkeypatch.setattr(lifecycle, 'validate', lambda options: None)
    monkeypatch.setattr(response_runtime, 'verify_runtime', lambda options: {})
    monkeypatch.setattr(native_assets, 'verify_native', lambda options: {})
    before = torch.get_rng_state().clone()
    states = [g.get_state().clone() for g in generators]
    expected = torch.rand(3, generator=torch.Generator().manual_seed(17))
    def construct(options, *, runtime=None):
        assert torch.equal(torch.rand(3), expected)
        assert torch.equal(torch.rand(3, generator=generators[1]), expected)
        if fail:
            raise RuntimeError('provider/critic/game construction failed')
        return 'complete binding'
    monkeypatch.setattr(lifecycle, '_prepare', construct)
    if fail:
        with pytest.raises(RuntimeError, match='provider/critic/game'):
            lifecycle.prepare({'device': 'cuda:1', 'seed': 17})
    else:
        assert lifecycle.prepare({'device': 'cuda:1', 'seed': 17}) == 'complete binding'
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(g.get_state(), state) for g, state in zip(generators, states))
    assert visited == [('get', 1), ('set', 1)]


def test_native_prepare_refuses_preflight_before_forking_or_constructing(monkeypatch, cuda_generators):
    from yue2_sliders import lifecycle, native_assets, response_runtime
    _, visited = cuda_generators
    monkeypatch.setattr(lifecycle, 'validate', lambda options: None)
    monkeypatch.setattr(response_runtime, 'verify_runtime', lambda options: {})
    def refuse(options): raise ValueError('changed strict native preflight')
    monkeypatch.setattr(native_assets, 'verify_native', refuse)
    monkeypatch.setattr(lifecycle, '_prepare', forbidden)
    with pytest.raises(ValueError, match='changed strict native preflight'):
        lifecycle.prepare({'device': 'cuda:1'})
    assert visited == []


@pytest.mark.parametrize('domain', ['auto', 'continuous'])
def test_fresh_process_native_factory_scope_initializes_same_real_game_and_exact_continuation(domain):
    """Replace only the pretrained boundary with the existing random CPU host.

    The native prepare branch constructs the actual provider, audio critic,
    shared prior/streams and PG optimizers. This is CPU wiring evidence only.
    Before the outer scope fix the critic differs across these caller seeds.
    """
    import json, os, subprocess, sys
    source_profile = 'release-0.9.5'
    from particle_sliders.pg09 import _released_sources
    try:
        _released_sources(source_profile)
    except RuntimeError:
        pytest.skip('requires its separately installed exact ' + source_profile + ' cohort')
    code = r'''
from copy import deepcopy
import json, os
import torch
from yue2_sliders import lifecycle, native_assets
from yue2_sliders.response_native import build_native as actual_build
from particle_sliders.pg09 import _hash
# The fixture boundary has the same caller-RNG-preserving behavior as the
# genuine native loader. Do not inherit dummy's historical global seed.
def fixture_boundary(options, **kwargs):
    with torch.random.fork_rng(devices=[]):
        return actual_build(dict(options, dummy=True, device='cpu', base_precision='float32'), **kwargs)
lifecycle.build_native = fixture_boundary
native_assets.verify_native = lambda options: {'fixture_boundary_only': True}
torch.random.default_generator.manual_seed(int(os.environ['CALLER_SEED']))
torch.rand(37)
before = torch.get_rng_state().clone()
options = dict(device='cpu', seed=7, steps=2, batch=2, sample_seeds=1, history_tokens=4,
    pg_source_profile=os.environ['TEST_PG_SOURCE_PROFILE'],
    penalty_input_domain=os.environ['TEST_PG_DOMAIN'],
    rows=[dict(neutral='soft piano', positive='bright piano', lyrics='A new day begins'),
          dict(neutral='quiet flute', positive='strong flute', lyrics='The river carries light')])
binding = lifecycle.prepare(options)
assert torch.equal(before, torch.get_rng_state())
try:
    game = binding.game
    initial = game.state_dict()
    keys = ('network','critic','prior','spread','streams','row_orders','row_cursors',
            'exposures','model_cpu_rng','model_cuda_rng','config')
    hashes = {key: _hash(initial[key]) for key in keys}
    first = game.step(); one = deepcopy(game.state_dict())
    second = game.step(); final = _hash(game.state_dict())
    game.load_state_dict(one)
    assert game.step() == second and _hash(game.state_dict()) == final
finally:
    binding.close()
assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
# Negative control reproduces the old unscoped critic construction while the
# pretrained boundary still preserves callers. It must expose the caller seed.
with torch.random.fork_rng(devices=[]):
    unscoped = lifecycle._prepare(options)
    try: unscoped_critic = _hash(unscoped.game.state_dict()['critic'])
    finally: unscoped.close()
print(json.dumps(dict(initial=hashes, first=first, second=second, final=final,
    caller_CPU_RNG_restored=True, exact_2_vs_1_plus_1=True, CUDA_initialized=False,
    unscoped_critic=unscoped_critic),sort_keys=True))
'''
    outputs = []
    for caller, hashseed in [('1234', '19'), ('98765', '403')]:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONHASHSEED=hashseed,
                           CALLER_SEED=caller, TEST_PG_SOURCE_PROFILE=source_profile, TEST_PG_DOMAIN=domain,
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        run = subprocess.run([sys.executable, '-c', code], env=environment,
                             text=True, capture_output=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
        outputs.append(json.loads(run.stdout.strip().splitlines()[-1]))
    assert outputs[0].pop('unscoped_critic') != outputs[1].pop('unscoped_critic')
    assert outputs[0] == outputs[1]


def test_native_identity_binds_factory_provider_and_history_source_bytes():
    from pathlib import Path
    import hashlib
    from yue2_sliders import response_runtime
    sources = response_runtime.native_sources()
    root = Path(response_runtime.__file__).parent
    for name in ('lifecycle.py', 'response_provider.py', 'train.py'):
        assert sources[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
