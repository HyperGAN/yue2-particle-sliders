"""Per-token RAW strength on the resident YuE2 eager semantic stage."""

from contextlib import ExitStack
from math import isfinite
from sys import float_info


def generate(pipeline, plan, *, scale_adapter, step_gains, strength=1.0,
             sampling=None, cancelled=None, on_token=None):
    """Generate native semantic codes and return the strengths used through EOS."""
    from yue2.protocol import resolve_sampling

    if (pipeline.backend != "torch-eager" or pipeline.quantization != "none"
            or pipeline.offload_ar or pipeline._model is None):
        raise ValueError("YuE2 requires its resident eager RAW adapter and AR model")
    if plan.request.cot != "off" or plan.request.guidance != 1:
        raise ValueError("YuE2 scheduling requires the single-branch CoT-off semantic path")
    budget = resolve_sampling(sampling, pipeline.generation_config.semantic).max_tokens
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != budget:
        raise ValueError("YuE2 requires one gain per possible semantic output token")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("nominal YuE2 strength must be finite in [0,1]")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("YuE2 token gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not isfinite(value) or value > 1 for value in strengths):
        raise ValueError("effective YuE2 strengths must be finite in [0,1]")

    model = pipeline._model
    prefix = len(plan.prefix)
    scope = ExitStack()
    applied = []

    def before(_module, args, inputs):
        position = len(applied)
        ids = args[0] if args else inputs.get("input_ids")
        cache = inputs.get("past_key_values")
        width = prefix if position == 0 else 1
        expected_length = 0 if position == 0 else prefix + position - 1
        if (position >= budget or ids is None or ids.shape != (1, width)
                or cache is None or int(cache.get_seq_length()) != expected_length):
            raise RuntimeError("YuE2 cached position differs from the requested schedule")
        value = strengths[position]
        scope.enter_context(scale_adapter(value))
        applied.append(value)

    def after(_module, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(model.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        semantic = pipeline.generate_semantic(plan, sampling=sampling, cancelled=cancelled, on_token=on_token)
    if len(applied) != semantic.timing["output_tokens"]:
        raise RuntimeError("YuE2 semantic token count differs from the applied schedule")
    return semantic, applied
