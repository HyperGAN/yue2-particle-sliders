"""Base+adapter-only AR response inference; never constructs training components."""
from functools import partial

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg09_audio import YuE2ARResponseAdapter, require_audio_width

from .response_runtime import (PG_VERSION, PROFILE, DOMAIN, PROTOCOL, capabilities,
    validate_native_options, validate_staged, verify_runtime, native_sources, frozen_digest, restore_model, observation)

_RUNTIME = {'device', 'revision', 'cache_dir'}
_CONFIGURATION = {'dummy', 'base_seed', 'base_precision', 'architecture',
                  'pg_version', 'pg_source_profile', 'penalty_input_domain', 'context_limit'}


def check_tokens(ids, architecture, context_limit):
    if not isinstance(ids, (list, tuple)) or not ids or any(type(i) is not int or not 0 <= i < architecture['vocab_size'] for i in ids):
        raise ValueError('token_ids must contain native in-vocabulary integer tokens')
    if len(ids) > context_limit:
        raise ValueError('token_ids exceed the native context limit')


class NativeResponse(YuE2ARResponseAdapter):
    def response(self, *, token_ids, strength=1., seed=0):
        from .backend import architecture_of
        check_tokens(token_ids, architecture_of(self.model), int(self.model.config.max_position_embeddings))
        # The outer inference session owns RNG; hidden() enforces the context limit.
        return self._response(tuple(token_ids), scale=strength).detach().clone()


def build_native(options, *, runtime=None, checkpoint_training=False):
    """Own the existing loader/slider/AR adapter and unwind partial setup."""
    validate_native_options(options)
    dependencies = verify_runtime()
    width = capabilities(options).response_shape[1]
    require_audio_width(width, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
    if runtime is not None:
        validate_staged(runtime, options)
    import torch
    from .backend import YuE2Backend, architecture_of
    from .slider import ParticleSlider
    from .stamp import locked_stamp
    torch.manual_seed(options.get('base_seed', 7))
    if runtime is None:
        runtime = YuE2Backend(device=options.get('device', 'cpu'), allow_hub=False,
            dummy=options.get('dummy', False), revision=options.get('revision'), cache_dir=options.get('cache_dir'))
    validate_staged(runtime, options)
    device = torch.device(options.get('device', 'cpu'))
    restore = restore_model(runtime.model)
    try:
        identity = dict(base=runtime.identity, frozen_base_sha256=frozen_digest(runtime.model),
                        dependencies=dependencies, product_sources=native_sources(), base_seed=options.get('base_seed', 7))
        slider = ParticleSlider(runtime.model, locked_stamp()[0])
        native = NativeResponse(runtime, slider, model_identity=identity,
            checkpoint_modules=list(runtime.model.model.layers) if checkpoint_training and not runtime.dummy else (),
            pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
        native.inference_configuration = dict(dummy=runtime.dummy, base_seed=options.get('base_seed', 7),
            base_precision='float32' if device.type == 'cpu' else 'bfloat16', architecture=architecture_of(runtime.model),
            pg_version=PG_VERSION, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN,
            context_limit=int(runtime.model.config.max_position_embeddings))
        return native, restore, partial(observation, runtime)
    except BaseException:
        restore()
        raise


def _native_options(options, configuration):
    return dict(configuration, **options)


def validate_inference(options, spec):
    config = spec.configuration
    if set(options) - _RUNTIME or set(config) != _CONFIGURATION:
        raise ValueError('native-only runtime accepts device/revision/cache_dir, never training rows or game options')
    if config['base_precision'] != 'float32' and 'device' not in options:
        raise ValueError('native device must be explicit')
    validate_native_options(_native_options(options, config))
    if type(config['context_limit']) is not int or config['context_limit'] < 1:
        raise ValueError('invalid native context limit')
    arch = config['architecture']
    if not isinstance(arch, dict) or any(type(v) is not int or v < 1 for v in arch.values()):
        raise ValueError('invalid native architecture descriptor')
    if arch.get('hidden_size') != capabilities(config).response_shape[1] or 'vocab_size' not in arch:
        raise ValueError('native architecture differs from the response descriptor')


def preflight_inference(options, request, spec):
    validate_inference(options, spec)
    config = spec.configuration
    require_audio_width(capabilities(config).response_shape[1], pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
    native = request['native']
    if native:
        if set(native) != {'token_ids'}:
            raise ValueError('numerical AR response requests require only token_ids')
        check_tokens(native['token_ids'], config['architecture'], config['context_limit'])


def prepare_inference(options, spec):
    native, close, observe = build_native(_native_options(options, spec.configuration))
    try:
        if native.inference_configuration != spec.configuration:
            raise ValueError('loaded native architecture/precision differs from the export')
        return InferenceBinding(native.network,
            native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL),
            close, response=native.response, observation=observe)
    except BaseException:
        close()
        raise


INFERENCE = InferenceBackend('yue2', capabilities({'dummy': True}), prepare_inference,
    validate_inference, configure_capabilities=capabilities, preflight=preflight_inference)
