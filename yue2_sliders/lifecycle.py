"""Complete owned YuE2 numerical response factory; PG owns every GAN update."""
from particle_sliders.lifecycle import Binding
from particle_sliders.response import ResponseBackend

from .response_runtime import PG_VERSION, PROFILE, DOMAIN, PROTOCOL, capabilities, validate_native_options, validate_staged, engine_options
from .response_native import build_native, native_initialization_rng, INFERENCE

_ALLOWED = {'rows', 'dummy', 'device', 'base_precision', 'steps', 'seed', 'batch',
            'sample_seeds', 'history_tokens', 'preset', 'sampling', 'ema_decay',
            'revision', 'cache_dir', 'pg_version', 'pg_source_profile', 'penalty_input_domain',
            'lr', 'd_lr_mult', 'prior_lr_mult', 'native_numerical_profile'}


def _rows(options):
    from .prompts import parse_prompts
    return parse_prompts(options.get('rows'))[0]


def _native_options(options):
    return {**{key: options[key] for key in ('dummy', 'device', 'base_precision', 'revision', 'cache_dir',
        'pg_version', 'pg_source_profile', 'penalty_input_domain', 'native_numerical_profile') if key in options}, 'base_seed': options.get('seed', 7)}


def validate(options):
    if set(options) - _ALLOWED:
        raise ValueError('unsupported YuE2 response options: ' + str(sorted(set(options) - _ALLOWED)))
    from .learning_rates import recipe_overrides
    recipe_overrides(**{key: options.get(key) for key in ('lr', 'd_lr_mult', 'prior_lr_mult')})
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
    if options.get('dummy', False):
        return _prepare(options, runtime=runtime)
    # Refuse numerical/source/asset changes before a CUDA fork can initialize
    # the requested device. build_native independently rechecks its own entry.
    from .response_runtime import verify_runtime
    from .native_assets import verify_native
    native_options = _native_options(options)
    verify_runtime(native_options)
    verify_native(native_options)
    # The native base/adapter loader preserves the caller's RNG. The provider,
    # critic and game must also have an explicit seed, rather than inheriting
    # whichever caller state happened to precede a fresh reconstruction.
    with native_initialization_rng(native_options):
        return _prepare(options, runtime=runtime)


def _prepare(options, *, runtime=None):
    from .learning_rates import recipe_overrides
    rates = recipe_overrides(**{key: options.get(key) for key in ('lr', 'd_lr_mult', 'prior_lr_mult')})
    native, restore, observe = build_native(_native_options(options), runtime=runtime, checkpoint_training=True)
    provider = None
    closed = False
    def close():
        nonlocal closed
        if not closed:
            try:
                if provider is not None:
                    provider.close()
            finally:
                restore()
            closed = True
    try:
        from .response_provider import PreparedBank
        from particle_sliders.pg095 import ConditionalPG095 as ConditionalPG09
        from particle_sliders.pg09_audio import AudioResponseCritic
        import torch
        provider = PreparedBank(native.backend, native, _rows(options),
            sample_seeds=options.get('sample_seeds', 1), history_tokens=options.get('history_tokens', 4), seed=options.get('seed', 7))
        critic = AudioResponseCritic(native.model.config.hidden_size, provider.condition_width).to(
            device=next(native.network.parameters()).device, dtype=torch.float32)
        game = ConditionalPG09(native, provider, critic, particles=native.particles,
            response_domain='continuous_vector', preset=options.get('preset', 'bispf_optimized'),
            batch_size=options.get('batch', 2), seed=options.get('seed', 7), total_steps=options.get('steps', 4),
            sampling=options.get('sampling', 'balanced'), recipe_overrides=rates, **engine_options(options))
        spec = INFERENCE.specification(native, model_identity=native.model_identity,
            protocol=PROTOCOL, configuration=native.inference_configuration)
        before_update = None
        if not native.backend.dummy:
            def before_update(context):
                from .native_assets import numerical_profile
                if numerical_profile(_native_options(options), native.model) != native.model_identity['native_numerical_profile']:
                    raise ValueError('native YuE2 current numerical identity differs before update')
                return True
        return Binding(game, close, inference=spec, observation=observe, before_update=before_update)
    except BaseException:
        close()
        raise


BACKEND = ResponseBackend('yue2', capabilities({'dummy': True}), prepare, validate,
                          configure_capabilities=capabilities, preflight=preflight)
