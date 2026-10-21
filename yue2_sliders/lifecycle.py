"""Complete owned YuE2 numerical response factory; PG owns every GAN update."""
from particle_sliders.lifecycle import Binding
from particle_sliders.response import ResponseBackend

from .response_runtime import PG_VERSION, PROFILE, DOMAIN, PROTOCOL, capabilities, validate_native_options, validate_staged
from .response_native import build_native, INFERENCE

_ALLOWED = {'rows', 'dummy', 'device', 'base_precision', 'steps', 'seed', 'batch',
            'sample_seeds', 'history_tokens', 'preset', 'sampling', 'ema_decay',
            'revision', 'cache_dir', 'pg_version', 'pg_source_profile', 'penalty_input_domain'}


def _rows(options):
    from .prompts import parse_prompts
    return parse_prompts(options.get('rows'))[0]


def _native_options(options):
    return {**{key: options[key] for key in ('dummy', 'device', 'base_precision', 'revision', 'cache_dir',
        'pg_version', 'pg_source_profile', 'penalty_input_domain') if key in options}, 'base_seed': options.get('seed', 7)}


def validate(options):
    if set(options) - _ALLOWED:
        raise ValueError('unsupported YuE2 response options: ' + str(sorted(set(options) - _ALLOWED)))
    rows = _rows(options)
    validate_native_options(_native_options(options))
    for key, default, low in [('steps', 4, 1), ('seed', 7, 0), ('batch', 2, 2),
                             ('sample_seeds', 1, 1), ('history_tokens', 4, 1)]:
        if type(options.get(key, default)) is not int or options.get(key, default) < low:
            raise ValueError(key + ' must be an integer within its response budget')
    if not 2 <= len(rows) * options.get('sample_seeds', 1) <= 512:
        raise ValueError('prepared prompt/history bank must contain 2..512 rows')
    if options.get('batch', 2) > len(rows) * options.get('sample_seeds', 1):
        raise ValueError('batch exceeds prepared prompt/history bank')
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('unsupported public PG preset')
    if options.get('sampling', 'balanced') not in ('balanced', 'replacement'):
        raise ValueError('unsupported shared sampler')


def preflight(options, request, staged):
    if request['native']:
        raise ValueError('v1 evaluation selects a prepared row; new token_ids belong to native-only v2 response inference')
    if set(staged) - {'runtime'}:
        raise ValueError('unknown staged runtime object')
    runtime = staged.get('runtime')
    if runtime is not None:
        validate_staged(runtime, _native_options(options))


def prepare(options, *, runtime=None):
    validate(options)
    native, restore, observe = build_native(_native_options(options), runtime=runtime, checkpoint_training=True)
    provider = None
    closed = False
    def close():
        nonlocal closed
        if not closed:
            closed = True
            try:
                if provider is not None:
                    provider.close()
            finally:
                restore()
    try:
        from .response_provider import PreparedBank
        from particle_sliders import ConditionalPG09
        from particle_sliders.pg09_audio import AudioResponseCritic
        import torch
        provider = PreparedBank(native.backend, native, _rows(options),
            sample_seeds=options.get('sample_seeds', 1), history_tokens=options.get('history_tokens', 4), seed=options.get('seed', 7))
        critic = AudioResponseCritic(native.model.config.hidden_size, provider.condition_width).to(
            device=next(native.network.parameters()).device, dtype=torch.float32)
        game = ConditionalPG09(native, provider, critic, particles=native.particles,
            response_domain='continuous_vector', preset=options.get('preset', 'bispf_optimized'),
            batch_size=options.get('batch', 2), seed=options.get('seed', 7), total_steps=options.get('steps', 4),
            sampling=options.get('sampling', 'balanced'), pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
        spec = INFERENCE.specification(native, model_identity=native.model_identity,
            protocol=PROTOCOL, configuration=native.inference_configuration)
        return Binding(game, close, inference=spec, observation=observe)
    except BaseException:
        close()
        raise


BACKEND = ResponseBackend('yue2', capabilities({'dummy': True}), prepare, validate,
                          configure_capabilities=capabilities, preflight=preflight)
