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
