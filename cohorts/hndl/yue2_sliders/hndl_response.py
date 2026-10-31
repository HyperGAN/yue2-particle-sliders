"""Opt-in YuE2 network cohort using the real shared HyperGAN/HNDL builder.

The shared core owns every GAN operation and session/checkpoint lifecycle.
The full-width native AR domain is still refused before loading model weights.
Install this candidate in its own exact environment; old packets are not migrated.
"""
from dataclasses import replace
from functools import partial
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import torch
from torch import nn
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding
from particle_sliders.response import ResponseBackend

from . import response_runtime as legacy
from .response_native import NativeOwner, NativeResponse, NativeCleanupError, check_tokens

CORE_PIN = '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
PG_VERSION = '0.9.4'
PROFILE = 'release-0.9.4'
DOMAIN = 'auto'
PROTOCOL = 'yue2-hypergan-hndl-ar-hidden-response-v1'
ROOT = Path(__file__).parent


def verify_runtime():
    lock = json.loads((ROOT / 'hndl-runtime.lock.json').read_text())
    if (lock['core_commit'], lock['core_version'], lock['pg_version']) != (CORE_PIN, '0.12.0', PG_VERSION):
        raise ValueError('YuE2 HNDL code and source lock disagree')
    distribution = metadata.distribution('particle-sliders-core')
    if distribution.version != lock['core_version']:
        raise ValueError('YuE2 HNDL requires its separate core0.12 cohort; retain the original environment for old packets')
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    archive_hash = direct.get('archive_info', {}).get('hashes', {}).get('sha256')
    vcs = direct.get('vcs_info', {})
    if 'archive_info' in direct:
        if archive_hash != lock['core_wheel_sha256']:
            raise ValueError('YuE2 HNDL core wheel SHA256 differs from the exact candidate')
    elif (vcs.get('vcs'), vcs.get('commit_id'), direct.get('url', '').removesuffix('.git')) != (
            'git', CORE_PIN, 'https://github.com/HyperGAN/particle-sliders'):
        raise ValueError('YuE2 HNDL requires exact wheel archive or Git commit provenance')
    if 'particlegan==0.9.4' not in {r.replace(' ', '') for r in distribution.requires or ()}:
        raise ValueError('YuE2 HNDL core must pin exact ParticleGAN0.9.4')
    root = Path(distribution.locate_file('')).resolve()
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for package in ('particle_sliders', 'concept_slider_core')
              for p in sorted((root / package).rglob('*'))
              if p.is_file() and (p.suffix in ('.py', '.json', '.hndl') or p.name == 'LICENSE')}
    if actual != lock['source_sha256']:
        raise ValueError('YuE2 HNDL shared source closure differs from the exact candidate')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('particle_sliders', 'concept_slider_core'):
            source = getattr(module, '__file__', None)
            if source is None or not Path(source).resolve().is_relative_to(root) or str(Path(source).resolve().relative_to(root)) not in actual:
                raise ValueError('unqualified shared import origin: ' + name)
    from particle_sliders.pg09 import _released_sources
    from particle_sliders.fibo_hndl import verified_builder
    if metadata.version('particlegan') != PG_VERSION or _released_sources(PROFILE) != lock['pg_sources_sha256']:
        raise ValueError('YuE2 HNDL PG release sources differ')
    builder, identity = verified_builder()  # Actual HyperGAN develop/HNDL0.10 closure.
    return builder, dict(core_commit=CORE_PIN, **identity)


def capabilities(options):
    return replace(legacy.capabilities(options), pg_version=PG_VERSION, pg_source_profile=PROFILE,
        qualification='Opt-in HyperGAN/HNDL CPU fixture cohort; pretrained/GPU/acoustic execution unqualified')


def validate_native(options):
    profile = (options.get('pg_version', PG_VERSION), options.get('pg_source_profile', PROFILE),
               options.get('penalty_input_domain', DOMAIN))
    if profile != (PG_VERSION, PROFILE, DOMAIN):
        raise ValueError('YuE2 HNDL requires exact PG0.9.4 release-0.9.4/auto; no domain override')
    legacy.validate_native_options({k: v for k, v in options.items()
                                   if k not in {'pg_version', 'pg_source_profile', 'penalty_input_domain'}})


class HNDLSlider(nn.Module):
    """YuE2 projection selection; shared HNDLParticleProjection owns network math."""
    def __init__(self, model):
        super().__init__()
        from particle_sliders.fibo_hndl import HNDLParticleProjection
        from .backend import attention_targets
        from .slider import RoutedProjection
        from .stamp import locked_stamp
        targets = attention_targets(model)
        if any(isinstance(getattr(m.forward, '__self__', None), (RoutedProjection, HNDLParticleProjection))
               for m in targets.values()):
            raise ValueError('A YuE2 slider is already attached')
        builder, identity = verify_runtime()  # Verify before hooks or trainability changes.
        stamp, _ = locked_stamp()
        rank, alpha = int(stamp.spec['adapter_rank']), float(stamp.spec['adapter_alpha'])
        shape = (int(stamp.spec['parts']), int(stamp.spec['particle_dim']))
        if (rank, alpha, shape) != (8, 8., (128, 4)):
            raise ValueError('YuE2 HNDL source cohort requires the shared rank8/alpha8/128x4 geometry')
        model.requires_grad_(False)
        device = next(model.parameters()).device
        self.particles = nn.Parameter(torch.randn(*shape, device=device))
        self.adapters = nn.ModuleDict()
        for path, module in targets.items():
            key = path.replace('.', '-')
            self.adapters[key] = HNDLParticleProjection(key, module, self.particles, rank, alpha,
                                                        builder=builder, identity=identity)
        self.to(device=device, dtype=torch.float32)

    def scaled(self, strength):
        from .slider import _Scale
        return _Scale(self, strength)


class HNDLAudioCritic(nn.Module):
    """Native hidden-vector raw-logit readout built by HyperGAN, without reshaping."""
    def __init__(self, width, condition_width):
        super().__init__()
        builder, identity = verify_runtime()
        source = (ROOT / 'hndl_audio_critic.hndl').read_text()
        self.graph = builder(source, input_shape={'response': ['B', width], 'condition': ['B', condition_width]},
                             output_shape=['B', 1])
        self._identity = dict(identity, source_sha256=hashlib.sha256(source.encode()).hexdigest())

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)


def build_native(options, *, runtime=None, checkpoint_training=False):
    validate_native(options)
    from particle_sliders.pg09_audio import require_audio_width
    require_audio_width(capabilities(options).response_shape[1], pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
    _, dependencies = verify_runtime()
    from .backend import YuE2Backend, architecture_of
    if runtime is not None:
        legacy.validate_staged(runtime, options)
    torch.manual_seed(options.get('base_seed', 7))
    if runtime is None:
        runtime = YuE2Backend(device=options.get('device', 'cpu'), allow_hub=False,
            dummy=options.get('dummy', False), revision=options.get('revision'), cache_dir=options.get('cache_dir'))
    legacy.validate_staged(runtime, options)
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    owner = NativeOwner(runtime.model, slider_factory=HNDLSlider, projection_type=HNDLParticleProjection)
    try:
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [ROOT / 'hndl_response.py', ROOT / 'hndl_audio_critic.hndl', ROOT / 'hndl-runtime.lock.json']}
        identity = dict(base=runtime.identity, frozen_base_sha256=legacy.frozen_digest(runtime.model),
            dependencies=dependencies, product_sources=dict(legacy.native_sources(), **sources),
            base_seed=options.get('base_seed', 7))
        slider = owner.attach_slider(runtime.model)
        native = NativeResponse(runtime, slider, model_identity=identity,
            checkpoint_modules=list(runtime.model.model.layers) if checkpoint_training and not runtime.dummy else (),
            pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
        owner.adapter = native
        native.inference_configuration = dict(dummy=runtime.dummy, base_seed=options.get('base_seed', 7),
            base_precision='float32' if options.get('device', 'cpu') == 'cpu' else 'bfloat16',
            architecture=architecture_of(runtime.model), context_limit=int(runtime.model.config.max_position_embeddings),
            pg_version=PG_VERSION, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
        return native, owner.close, partial(legacy.observation, runtime)
    except BaseException as setup_error:
        try:
            owner.close()
        except BaseException as cleanup_error:
            raise NativeCleanupError(owner, setup_error, cleanup_error) from setup_error
        raise


def _native_options(options):
    from .lifecycle import _native_options as old_options
    return old_options(options)


def validate(options):
    from .lifecycle import validate as old_validate
    validate_native(_native_options(options))
    old_validate({k: v for k, v in options.items() if k not in {'pg_version', 'pg_source_profile', 'penalty_input_domain'}})


def prepare(options, *, runtime=None):
    validate(options)
    native, restore, observe = build_native(_native_options(options), runtime=runtime, checkpoint_training=True)
    provider = None
    def close():
        try:
            if provider is not None:
                provider.close()
        finally:
            restore()
    try:
        from .lifecycle import _rows
        from .response_provider import PreparedBank
        from particle_sliders import ConditionalPG09
        provider = PreparedBank(native.backend, native, _rows(options), sample_seeds=options.get('sample_seeds', 1),
            history_tokens=options.get('history_tokens', 4), seed=options.get('seed', 7))
        critic = HNDLAudioCritic(native.model.config.hidden_size, provider.condition_width).to(
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


def validate_inference(options, spec):
    from .response_native import _RUNTIME, _CONFIGURATION
    config = spec.configuration
    if set(options) - _RUNTIME or set(config) != _CONFIGURATION:
        raise ValueError('YuE2 HNDL inference accepts only native runtime options')
    merged = dict(config, **options)
    validate_native(merged)
    if config['base_precision'] != 'float32' and 'device' not in options:
        raise ValueError('native device must be explicit')
    arch = config['architecture']
    if (type(config['context_limit']) is not int or config['context_limit'] < 1 or
            not isinstance(arch, dict) or any(type(v) is not int or v < 1 for v in arch.values()) or
            arch.get('hidden_size') != capabilities(config).response_shape[1] or 'vocab_size' not in arch):
        raise ValueError('invalid native architecture/context descriptor')


def preflight_inference(options, request, spec):
    validate_inference(options, spec)
    from particle_sliders.pg09_audio import require_audio_width
    config = spec.configuration
    require_audio_width(capabilities(config).response_shape[1], pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
    if request['native']:
        if set(request['native']) != {'token_ids'}:
            raise ValueError('numerical AR response requests require only token_ids')
        check_tokens(request['native']['token_ids'], config['architecture'], config['context_limit'])


def prepare_inference(options, spec):
    native, close, observe = build_native(dict(spec.configuration, **options))
    try:
        if native.inference_configuration != spec.configuration:
            raise ValueError('native architecture/precision differs from export')
        return InferenceBinding(native.network,
            native_identity(native, model_identity=native.model_identity, protocol=PROTOCOL),
            close, response=native.response, observation=observe)
    except BaseException:
        close()
        raise


def preflight(options, request, staged):
    if request['native']:
        raise ValueError('training evaluation selects a prepared row; use native-only inference for token_ids')
    if set(staged) - {'runtime'}:
        raise ValueError('unknown staged runtime object')
    if staged.get('runtime') is not None:
        legacy.validate_staged(staged['runtime'], _native_options(options))


BACKEND = ResponseBackend('yue2-hndl', capabilities({'dummy': True}), prepare, validate,
                          configure_capabilities=capabilities, preflight=preflight)
INFERENCE = InferenceBackend('yue2-hndl', capabilities({'dummy': True}), prepare_inference,
    validate_inference, configure_capabilities=capabilities, preflight=preflight_inference)
