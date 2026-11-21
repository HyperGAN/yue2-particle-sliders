import json
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from yue2_sliders.activation_gate import CORE_MODULE, CORE_REVISION, SemanticGatePlan
from yue2_sliders.scheduled_semantic import generate, generate_gated


@pytest.fixture
def core_wheel():
    wheel = os.environ.get("YUE2_TEST_STEERING_WHEEL")
    if wheel is None:
        pytest.skip("Set YUE2_TEST_STEERING_WHEEL to the verified core PR #269 wheel")
    return Path(wheel)


@pytest.fixture(params=[1, 2])
def native(request):
    protocol = pytest.importorskip("yue2.protocol")
    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
    from yue2.pipeline import YuE2Pipeline
    from yue2_sliders.slider import ParticleSlider
    from yue2_sliders.stamp import locked_stamp

    first, second = protocol.CODEC_OFFSET, protocol.CODEC_OFFSET + 1
    eos = protocol.MUSIC_END
    branches = request.param
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        config = YuE2Config(hidden_size=16, num_hidden_layers=1, intermediate_size=32,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                            vocab_size=first + protocol.CODEC_SIZE, max_latent_frames=8)
        model = YuE2ForCausalLM(config).eval()
        slider = ParticleSlider(model, locked_stamp()[0])
    with torch.no_grad():
        model.get_input_embeddings().weight[first, 0] = -1
        model.get_input_embeddings().weight[second, 0] = 1
        for adapter in slider.adapters.values():
            adapter.lora_up.weight.fill_(0.02)

    pipe = YuE2Pipeline.__new__(YuE2Pipeline)
    pipe.backend, pipe.quantization, pipe.offload_ar = "torch-eager", "none", False
    pipe._model, pipe.device, pipe.progress = model, torch.device("cpu"), False
    pipe.tokenizer = SimpleNamespace(encode=lambda _text: [4, 5, 6, 7])
    pipe.generation_config = SimpleNamespace(semantic=protocol.Sampling(
        temperature=0, repetition_penalty=1, min_tokens=0, max_tokens=4))
    cfg = 1.0 if branches == 1 else None
    plan = pipe.plan(request=protocol.SongRequest("crisp piano", "", cot="off", cfg_scale=cfg))
    logits, scales = [], []
    fixture = SimpleNamespace(pipe=pipe, plan=plan, slider=slider, branches=branches,
                              projection=[1.0] + [0.0] * 15, first=first, second=second,
                              eos=eos, logits=logits, scales=scales, forced=[first, second, eos], fail=False)

    def record_and_force(_module, _inputs, output):
        position = len(logits) // branches
        if fixture.fail and position == 1:
            raise RuntimeError("fixture interruption in active native forward")
        logits.append(output.clone())
        scale, = {part.multiplier for part in slider.adapters.values()}
        scales.append(scale)
        # Keep true native logits above. A controlled legal token stream makes
        # the feedback and EOS coordinates deterministic for the fixture.
        selected = torch.zeros_like(output)
        selected[..., fixture.forced[position]] = 1
        return selected

    handle = model.lm_head.register_forward_hook(record_and_force)
    try:
        yield fixture
    finally:
        handle.remove()


def test_native_semantic_gate_projects_feedback_once_and_replays(core_wheel, native):
    slider = native.slider
    gate_plan = SemanticGatePlan(core_wheel=core_wheel, method="cfp", parameter=0,
                                direction=native.projection, calibration_sha256="a" * 64)
    visited = []
    sample, trace = generate_gated(native.pipe, native.plan, scale_adapter=slider.scaled,
                                   strength=0.5, gate_plan=gate_plan,
                                   on_token=lambda _phase, token: visited.append(token))
    assert not sample.truncated and sample.tokens == [0, 1]
    assert visited == [native.first, native.second, native.eos]
    assert trace["activation_gate"]["core_revision"] == CORE_REVISION
    assert trace["applied_strengths"] == [0, 0.5, 0]
    assert trace["eligible"] == 2 and trace["fired"] == 1 and trace["total_strength"] == 0.5
    assert [row["prediction_index"] for row in trace["decisions"]] == [1, 2]
    assert [row["feedback_token"] for row in trace["decisions"]] == [native.first, native.second]
    assert [row["projection"] for row in trace["decisions"]] == [-1, 1]
    assert native.scales == [0] * native.branches + [0.5] * native.branches + [0] * native.branches
    adaptive_logits = native.logits.copy()
    native.logits.clear()

    replay, applied = generate(native.pipe, native.plan, scale_adapter=slider.scaled,
                               strength=0.5, step_gains=[0, 1, 0, 0])
    assert applied == trace["applied_strengths"] and replay.tokens == sample.tokens
    assert all(torch.equal(a, b) for a, b in zip(adaptive_logits, native.logits, strict=True))


def test_native_semantic_gate_off_clabs_eos_and_cleanup(core_wheel, native):
    slider = native.slider
    off = SemanticGatePlan(core_wheel=core_wheel, method="random", parameter=0,
                          seed=1, calibration_sha256="b" * 64)
    baseline = native.pipe.generate_semantic(native.plan)
    baseline_logits = native.logits.copy()
    native.logits.clear()
    with slider.scaled(0.25):
        result, trace = generate_gated(native.pipe, native.plan, scale_adapter=slider.scaled,
                                       strength=1, gate_plan=off)
        assert all(layer.multiplier == 0.25 for layer in slider.adapters.values())
    assert result.tokens == baseline.tokens and trace["applied_strengths"] == [0, 0, 0]
    assert all(torch.equal(a, b) for a, b in zip(baseline_logits, native.logits, strict=True))

    clabs = SemanticGatePlan(core_wheel=core_wheel, method="clabs", parameter=0.5, seed=1,
                            calibration_sha256="c" * 64, direction=native.projection,
                            clabs=dict(mean=0, std=1, epsilon=1e-6, setpoint=0,
                                       proportional_gain=1, integral_gain=0, beta=1))
    native.logits.clear()
    _, trace = generate_gated(native.pipe, native.plan, scale_adapter=slider.scaled,
                              strength=1, gate_plan=clabs)
    assert trace["applied_strengths"] == [0, 1, 0]
    json.dumps(trace, allow_nan=False)

    native.logits.clear()
    native.fail = True
    with slider.scaled(0.25), pytest.raises(RuntimeError, match="fixture interruption"):
        try:
            generate_gated(native.pipe, native.plan, scale_adapter=slider.scaled,
                           strength=1, gate_plan=clabs)
        finally:
            assert all(layer.multiplier == 0.25 for layer in slider.adapters.values())
            assert not native.pipe._model._forward_pre_hooks
            assert not native.pipe._model._forward_hooks

    native.fail = False
    native.forced[:] = [native.eos]
    native.logits.clear()
    result, trace = generate_gated(native.pipe, native.plan, scale_adapter=slider.scaled,
                                   strength=1, gate_plan=clabs)
    assert not result.truncated and not result.tokens
    assert trace["applied_strengths"] == [0]
    assert trace["eligible"] == 0 and trace["decisions"] == [] and trace["gate_rate"] is None


def test_gate_requires_exact_core_source(core_wheel, tmp_path):
    changed = tmp_path / "changed.whl"
    with ZipFile(core_wheel) as original, ZipFile(changed, "w") as archive:
        archive.writestr(CORE_MODULE, original.read(CORE_MODULE) + b"# changed\n")
    with pytest.raises(ValueError, match="exact shared steering module"):
        SemanticGatePlan(core_wheel=changed, method="random", parameter=0, seed=1,
                         calibration_sha256="a" * 64)
