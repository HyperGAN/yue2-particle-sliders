from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from yue2_sliders.scheduled_semantic import generate


def test_owned_native_semantic_token_schedule_uses_real_static_cache_and_eos():
    protocol = pytest.importorskip("yue2.protocol")
    pytest.importorskip("transformers")
    from yue2.sampling import generate_tokens

    class Model(torch.nn.Module):
        active = 0.0
        config = SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, head_dim=2)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, ids, *, past_key_values, use_cache, logits_to_keep):
            assert use_cache and logits_to_keep == 1
            keys = torch.zeros(1, 1, ids.shape[1], 2)
            past_key_values.update(keys, keys, 0)
            logits = torch.zeros(1, 1, protocol.CODEC_OFFSET + protocol.CODEC_SIZE)
            logits[..., protocol.CODEC_OFFSET] = 10
            logits[..., protocol.MUSIC_END] = 15 * self.active
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

        @contextmanager
        def scaled(self, value):
            previous, self.active = self.active, value
            try:
                yield
            finally:
                self.active = previous

    model = Model()
    sampling = protocol.Sampling(temperature=0, repetition_penalty=1.0, min_tokens=0, max_tokens=3)
    class Pipeline:
        backend = "torch-eager"
        quantization = "none"
        offload_ar = False
        _model = model
        generation_config = SimpleNamespace(semantic=sampling)

        def generate_semantic(self, plan, *, sampling, cancelled, on_token):
            tokens, timing, truncated = generate_tokens(
                model, plan.prefix, protocol.resolve_sampling(sampling, self.generation_config.semantic),
                21, "semantic", use_cuda_graph=False, legacy_off=True,
                cancelled=cancelled, on_token=on_token,
            )
            return SimpleNamespace(tokens=tokens, timing=timing, truncated=truncated)

    pipe = Pipeline()
    plan = SimpleNamespace(prefix=[1, 3, 5], request=SimpleNamespace(cot="off", guidance=1))
    with model.scaled(0.5):
        baseline = pipe.generate_semantic(plan, sampling=None, cancelled=None, on_token=None)
    constant, applied = generate(pipe, plan, scale_adapter=model.scaled, strength=0.5, step_gains=[1] * 3)
    assert baseline.tokens == constant.tokens and constant.truncated
    assert applied == [0.5] * 3
    visited = []
    varied, applied = generate(pipe, plan, scale_adapter=model.scaled, strength=0.5,
                               step_gains=[0.5, 1.5, 1], on_token=lambda _, token: visited.append(token))
    assert not varied.truncated and len(varied.tokens) == 1
    assert applied == [0.25, 0.75]
    assert visited[-1] == protocol.MUSIC_END and len(visited) == 2
    assert model.active == 0.0
