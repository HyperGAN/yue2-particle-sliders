"""Per-token RAW strength on the resident YuE2 eager semantic stage."""

from contextlib import ExitStack
from math import isfinite
from sys import float_info


def generate(pipeline, plan, *, scale_adapter, step_gains, strength=1.0,
             sampling=None, cancelled=None, on_token=None):
    """Generate native semantic codes and return the strengths used through EOS."""
    budget = _semantic_budget(pipeline, plan, sampling)
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != budget:
        raise ValueError("YuE2 requires one gain per possible semantic output token")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("nominal YuE2 strength must be finite in [0,1]")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("YuE2 token gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not isfinite(value) or value > 1 for value in strengths):
        raise ValueError("effective YuE2 strengths must be finite in [0,1]")

    def resolve(position, _ids):
        return strengths[position]

    return _run_semantic(pipeline, plan, scale_adapter, budget, resolve, sampling, cancelled, on_token)


def generate_gated(pipeline, plan, *, scale_adapter, gate_plan, strength,
                   sampling=None, cancelled=None, on_token=None):
    """Return native semantic codes and the trace of applied gate strengths."""
    from .activation_gate import SemanticGatePlan

    budget = _semantic_budget(pipeline, plan, sampling)
    if type(strength) not in (int, float) or not 0 < strength <= 1:
        raise ValueError("gated YuE2 requires an explicit nominal strength in (0,1]")
    if not isinstance(gate_plan, SemanticGatePlan):
        raise TypeError("gated YuE2 requires a calibrated SemanticGatePlan")

    project = gate_plan._projector(pipeline._model, strength)
    gate = gate_plan._new_gate()
    decisions = []

    def resolve(position, ids):
        if position == 0:
            # The initial prediction comes from a full prompt evaluation.
            # Leave that evaluation OFF and start with native token feedback.
            return 0.0
        projection = project(ids)
        decision = gate.step(projection)
        effective = float(strength) if decision.fired else 0.0
        decisions.append(dict(prediction_index=position, feedback_token=int(ids[0, 0]),
                              projection=projection, probability=decision.probability,
                              fired=decision.fired, effective_strength=effective))
        return effective

    semantic, applied = _run_semantic(pipeline, plan, scale_adapter, budget, resolve, sampling, cancelled, on_token)
    usage = gate.usage
    receipt = dict(activation_gate=gate_plan._receipt(), decisions=decisions,
                   applied_strengths=applied, nominal_strength=float(strength),
                   fired=usage.fired, eligible=usage.eligible,
                   gate_rate=usage.rate if usage.eligible else None,
                   total_strength=sum(applied))
    return semantic, receipt


def _semantic_budget(pipeline, plan, sampling):
    from yue2.protocol import resolve_sampling

    if (pipeline.backend != "torch-eager" or pipeline.quantization != "none"
            or pipeline.offload_ar or pipeline._model is None):
        raise ValueError("YuE2 requires its resident eager RAW adapter and AR model")
    if plan.request.cot != "off":
        raise ValueError("YuE2 scheduling requires the CoT-off semantic path")
    return resolve_sampling(sampling, pipeline.generation_config.semantic).max_tokens


def _run_semantic(pipeline, plan, scale_adapter, budget, resolve, sampling, cancelled, on_token):
    from yue2.protocol import negative_prefix

    model = pipeline._model
    prefix_lengths = [len(plan.prefix)]
    if plan.request.guidance != 1:
        prefix_lengths.append(len(negative_prefix(plan.request, pipeline.tokenizer)))
    branches = len(prefix_lengths)
    caches = []
    calls = 0
    scope = ExitStack()
    applied = []

    def before(_module, args, inputs):
        nonlocal calls
        position, branch = divmod(calls, branches)
        ids = args[0] if args else inputs.get("input_ids")
        cache = inputs.get("past_key_values")
        prefix = prefix_lengths[branch]
        width = prefix if position == 0 else 1
        expected_length = 0 if position == 0 else prefix + position - 1
        if (position >= budget or ids is None or ids.shape != (1, width)
                or cache is None or int(cache.get_seq_length()) != expected_length):
            raise RuntimeError("YuE2 cached position differs from the requested schedule")
        if position == 0:
            if any(cache is previous for previous in caches):
                raise RuntimeError("YuE2 guidance branches must own distinct caches")
            caches.append(cache)
        elif cache is not caches[branch]:
            raise RuntimeError("YuE2 cache differs from the expected guidance branch")
        if branch == 0:
            applied.append(resolve(position, ids))
        # Conditional and negative predictions share the decision for this
        # position while retaining their own native StaticKVCache instances.
        scope.enter_context(scale_adapter(applied[position]))
        calls += 1

    def after(_module, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(model.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        semantic = pipeline.generate_semantic(plan, sampling=sampling, cancelled=cancelled, on_token=on_token)
    if calls != len(applied) * branches or len(applied) != semantic.timing["output_tokens"]:
        raise RuntimeError("YuE2 semantic token count differs from the applied schedule")
    return semantic, applied
