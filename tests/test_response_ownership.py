"""Authored product ownership conformance; run only in the admitted installed slot."""
from contextlib import contextmanager

import pytest
import torch

from particle_sliders.pg09_backends import checkpoint_with_adapter_scale
from yue2_sliders.backend import YuE2Backend, attention_targets
from yue2_sliders.response_native import NativeOwner, NativeResponse
from yue2_sliders.slider import RoutedProjection


class OuterScale:
    scale = 1.

    @contextmanager
    def scaled(self, value):
        previous, self.scale = self.scale, value
        try:
            yield
        finally:
            self.scale = previous


def make_owned(seed=7, checkpointed=True, runtime=None):
    torch.manual_seed(seed)
    runtime = runtime or YuE2Backend(dummy=True)
    runtime.model.train()  # Broader native state must wait for all owned wrappers.
    owner = NativeOwner(runtime.model)
    try:
        slider = owner.attach_slider(runtime.model)
        with torch.no_grad():
            for adapter in slider.adapters.values():
                adapter.lora_up.weight.fill_(.02)
        blocks = [layer.self_attn.q_proj for layer in runtime.model.model.layers] if checkpointed else []
        native = NativeResponse(runtime, slider, model_identity={'fixture': 'owned tiny YuE2; no pretrained claim'},
                                checkpoint_modules=blocks)
        owner.adapter = native
        return runtime, native, owner, blocks
    except BaseException:
        owner.close()
        raise


def assert_detached(runtime):
    assert all('forward' not in vars(module) for module in attention_targets(runtime.model).values())
    assert not any(getattr(module, '_pg09_scale_checkpoint', False) for module in runtime.model.modules())
    assert runtime.model.training


@pytest.mark.parametrize('conflict', ['forward', 'marker'])
def test_checkpoint_conflict_retains_owner_closes_siblings_and_retries(conflict):
    runtime, native, owner, blocks = make_owned()
    block = blocks[0]
    wrapper = vars(block)['forward']
    if conflict == 'forward':
        foreign = lambda *args, **kwargs: None
        block.forward = foreign
    else:
        block._pg09_scale_checkpoint = False
    try:
        with pytest.raises(RuntimeError, match='ownership changed') as caught:
            owner.close()
        assert caught.value.owner is owner
        assert not owner.closed
        assert not runtime.model.training  # Broader restoration has not run.
        assert isinstance(getattr(blocks[1].forward, '__self__', None), RoutedProjection)
        assert not getattr(blocks[1], '_pg09_scale_checkpoint', False)
        if conflict == 'forward':
            assert block.forward is foreign
        else:
            assert block._pg09_scale_checkpoint is False
    finally:
        block.forward = wrapper
        block._pg09_scale_checkpoint = True
        owner.close()
    owner.close()
    native.close_checkpointing()
    assert_detached(runtime)


def test_projection_conflict_does_not_overwrite_external_owner_and_retry_is_safe():
    runtime, native, owner, blocks = make_owned()
    projection = runtime.model.model.layers[0].self_attn.k_proj
    owned = projection.forward
    foreign = lambda *args, **kwargs: None
    projection.forward = foreign
    try:
        with pytest.raises(RuntimeError, match='projection ownership changed') as caught:
            owner.close()
        assert caught.value.owner is owner
        assert projection.forward is foreign
        assert not owner.closed and not runtime.model.training
        # Checkpoint handles and successful projection siblings are already gone.
        assert all(not getattr(block, '_pg09_scale_checkpoint', False) for block in blocks)
        assert 'forward' not in vars(runtime.model.model.layers[1].self_attn.k_proj)
    finally:
        projection.forward = owned
        owner.close()
    assert_detached(runtime)


def test_nested_outer_handle_survives_product_cleanup_and_same_handle_nesting():
    runtime = YuE2Backend(dummy=True)
    block = runtime.model.model.layers[0].self_attn.v_proj
    handle = checkpoint_with_adapter_scale(block, OuterScale())
    with handle:
        with handle:
            outer_forward = block.forward
            runtime, native, owner, _ = make_owned(runtime=runtime)
            owner.close()
            assert block.forward is outer_forward
            assert block._pg09_scale_checkpoint is True
        assert block.forward is outer_forward  # Inner same-handle scope did not close it.
    assert_detached(runtime)
    with pytest.raises(RuntimeError, match='already closed'):
        handle.__enter__()


def test_conflicting_second_owner_does_not_detach_first_owner():
    runtime, native, owner, _ = make_owned()
    other = NativeOwner(runtime.model)
    with pytest.raises(ValueError, match='already attached'):
        other.attach_slider(runtime.model)
    other.close()
    assert any(isinstance(getattr(module.forward, '__self__', None), RoutedProjection)
               for module in attention_targets(runtime.model).values())
    owner.close()
    assert_detached(runtime)


def test_partial_projection_setup_unwinds_before_adapter_transfer(monkeypatch):
    runtime = YuE2Backend(dummy=True)
    runtime.model.train()
    owner = NativeOwner(runtime.model)
    original = RoutedProjection.__init__
    calls = 0
    def interrupted(self, *args, **kwargs):
        nonlocal calls
        original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError('fixture interruption after attachment')
    monkeypatch.setattr(RoutedProjection, '__init__', interrupted)
    with pytest.raises(RuntimeError, match='fixture interruption'):
        owner.attach_slider(runtime.model)
    owner.close()
    assert_detached(runtime)


@pytest.mark.parametrize('strength', [.5, 1.])
def test_deferred_backward_after_owner_close_retains_captured_scale(strength):
    def gradients(checkpointed):
        runtime, native, owner, _ = make_owned(checkpointed=checkpointed)
        with native.network.scaled(strength):
            loss = runtime.hidden([12, 21, 30])[:, -1, :3].sum()
        owner.close()  # The already-built graph still owns its captured forwards.
        loss.backward()
        return [None if p.grad is None else p.grad.clone() for p in native.network.parameters()]
    actual, expected = gradients(True), gradients(False)
    assert len(actual) == len(expected)
    assert any(value is not None and torch.count_nonzero(value) for value in actual)
    for left, right in zip(actual, expected):
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.equal(left, right)
