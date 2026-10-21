"""Base+adapter-only AR response inference; never constructs training components."""
from functools import partial

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg09_audio import YuE2ARResponseAdapter, require_audio_width

from .response_runtime import (PG_VERSION, PROFILE, DOMAIN, PROTOCOL, capabilities,
    validate_native_options, validate_staged, verify_runtime, native_sources, frozen_digest, observation)

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


class NativeCleanupError(RuntimeError):
    """Setup failed and cleanup needs conflict resolution; retain owner for retry."""
    def __init__(self, owner, setup_error, cleanup_error):
        super().__init__('YuE2 native cleanup is unresolved; resolve ownership and retry error.owner.close()')
        self.owner, self.setup_error, self.cleanup_error = owner, setup_error, cleanup_error


class NativeOwner:
    """Retain shared checkpoint ledgers; restore only owned projection forwards."""
    def __init__(self, model):
        from particle_sliders.lifecycle import native_cleanup
        from .backend import attention_targets
        self.model = model
        self.adapter = None
        self.closed = False
        self._attachment_attempted = False
        self._restore_native = native_cleanup(model)
        self._targets = [(module, 'forward' in vars(module), vars(module).get('forward'), module.forward)
                         for module in attention_targets(model).values()]
        self._projections = []

    def attach_slider(self, model):
        if model is not self.model:
            raise ValueError('native owner cannot attach a different model')
        if self.closed or self._attachment_attempted:
            raise RuntimeError('native owner permits one slider attachment attempt')
        self._attachment_attempted = True
        from .slider import ParticleSlider, RoutedProjection
        from .stamp import locked_stamp
        try:
            return ParticleSlider(model, locked_stamp()[0])
        finally:
            # Also claim projections installed before a partial constructor failure.
            for module, had_forward, previous, bound in self._targets:
                current = vars(module).get('forward')
                projection = getattr(current, '__self__', None)
                if isinstance(projection, RoutedProjection) and projection.org_forward == bound:
                    self._projections.append((module, current, had_forward, previous))

    def close(self):
        try:
            self._close_owned()
        except BaseException as error:
            # Common session/factory cleanup can fail before returning a Binding.
            # Keep the original shared error and an explicit retry owner together.
            error.owner = self
            raise

    def _close_owned(self):
        if self.closed:
            return
        # Shared handles check ownership, attempt all siblings, and retain failures.
        # Never restore broader owners while any checkpoint wrapper remains owned.
        if self.adapter is not None:
            self.adapter.close_checkpointing()
        first_error = None
        for index in range(len(self._projections) - 1, -1, -1):
            module, owned, had_forward, previous = self._projections[index]
            if vars(module).get('forward') is not owned:
                if first_error is None:
                    first_error = RuntimeError('YuE2 projection ownership changed; restore its owned forward before retrying cleanup')
                continue
            if had_forward:
                vars(module)['forward'] = previous
            else:
                vars(module).pop('forward', None)
            del self._projections[index]
        if first_error is not None:
            raise first_error
        self._restore_native()
        self.closed = True


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
    torch.manual_seed(options.get('base_seed', 7))
    if runtime is None:
        runtime = YuE2Backend(device=options.get('device', 'cpu'), allow_hub=False,
            dummy=options.get('dummy', False), revision=options.get('revision'), cache_dir=options.get('cache_dir'))
    validate_staged(runtime, options)
    device = torch.device(options.get('device', 'cpu'))
    owner = NativeOwner(runtime.model)
    try:
        identity = dict(base=runtime.identity, frozen_base_sha256=frozen_digest(runtime.model),
                        dependencies=dependencies, product_sources=native_sources(), base_seed=options.get('base_seed', 7))
        slider = owner.attach_slider(runtime.model)
        native = NativeResponse(runtime, slider, model_identity=identity,
            checkpoint_modules=list(runtime.model.model.layers) if checkpoint_training and not runtime.dummy else (),
            pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
        owner.adapter = native
        native.inference_configuration = dict(dummy=runtime.dummy, base_seed=options.get('base_seed', 7),
            base_precision='float32' if device.type == 'cpu' else 'bfloat16', architecture=architecture_of(runtime.model),
            pg_version=PG_VERSION, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN,
            context_limit=int(runtime.model.config.max_position_embeddings))
        return native, owner.close, partial(observation, runtime)
    except BaseException as setup_error:
        try:
            owner.close()
        except BaseException as cleanup_error:
            raise NativeCleanupError(owner, setup_error, cleanup_error) from setup_error
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
