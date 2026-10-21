"""Owned YuE2 AR response factory; never promises waveform or image output.

PG owns the game. Native tokenization, shared-history preparation and projection
hooks stay in this product. The legacy registry/schema name is a core seam to
rename; response metadata is the authoritative modality contract.
"""
import hashlib
import json
from pathlib import Path

from particle_sliders.lifecycle import Backend, Binding, Capabilities

CORE_PIN = 'fd89e1c780ab5210cd1be98ec3b70977e6349379'
LOCK = Path(__file__).with_name('lifecycle-runtime.lock.json')
_ALLOWED = {'rows', 'dummy', 'device', 'base_precision', 'steps', 'seed', 'batch',
            'sample_seeds', 'history_tokens', 'preset', 'sampling', 'ema_decay',
            'revision', 'cache_dir', 'pg_source_profile', 'penalty_input_domain'}


def _rows(options):
    from .prompts import parse_prompts
    return parse_prompts(options.get('rows'))[0]


def validate(options):
    """Reject unsupported operations/options without model or device allocation."""
    if set(options) - _ALLOWED:
        raise ValueError('unsupported YuE2 lifecycle options: ' + str(sorted(set(options) - _ALLOWED)))
    rows = _rows(options)
    for key, default, low in [('steps', 4, 1), ('seed', 7, 0), ('batch', 2, 2),
                              ('sample_seeds', 1, 1), ('history_tokens', 4, 1)]:
        if type(options.get(key, default)) is not int or options.get(key, default) < low:
            raise ValueError(key + ' must be an integer within its lifecycle budget')
    if options.get('batch', 2) > len(rows) * options.get('sample_seeds', 1):
        raise ValueError('batch exceeds prepared prompt/history bank')
    if type(options.get('dummy', False)) is not bool:
        raise ValueError('dummy must be a boolean')
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('unsupported public PG preset')
    if options.get('sampling', 'balanced') not in ('balanced', 'replacement'):
        raise ValueError('unsupported shared sampler')
    device = options.get('device', 'cpu')
    if not isinstance(device, str) or not (device == 'cpu' or device.startswith('cuda:')):
        raise ValueError('device must be cpu or an explicitly admitted cuda device')
    precision = 'float32' if device == 'cpu' else 'bfloat16'
    if options.get('dummy') and device != 'cpu':
        raise ValueError('dummy is CPU only')
    if options.get('base_precision', precision) != precision:
        raise ValueError('YuE2 native loader precision differs from requested base_precision')
    # No alternative source profile, projection or width truncation is selected.
    from particle_sliders.pg09_audio import require_audio_width
    if options.get('dummy'):
        require_audio_width(32, **_domain(options))


def _domain(options):
    return dict(pg_source_profile=options.get('pg_source_profile', 'release'),
                penalty_input_domain=options.get('penalty_input_domain', 'auto'))


def _restore_model(model):
    """Capture hooks before any partial projection/checkpoint construction."""
    keys = ('forward', 'gradient_checkpointing', '_gradient_checkpointing_func', '_pg09_scale_checkpoint')
    missing = object()
    saved = [(m, m.training, {key: m.__dict__.get(key, missing) for key in keys}) for m in model.modules()]
    gradients = [(p, p.requires_grad) for p in model.parameters()]
    def restore():
        for module, training, values in saved:
            module.training = training
            for key, value in values.items():
                if value is missing:
                    module.__dict__.pop(key, None)
                else:
                    setattr(module, key, value)
        for parameter, requires_grad in gradients:
            parameter.requires_grad_(requires_grad)
    return restore


def prepare(options, *, runtime=None):
    """Construct every owned game component; unwind all native hooks on failure.

    A staged runtime is for separately admitted integration tests and must expose
    the same frozen YuE2Backend interface. Callers never assemble PG components.
    """
    validate(options)
    from .integration import verify_lock
    dependencies = verify_lock(json.loads(LOCK.read_text()), CORE_PIN)
    import torch
    from particle_sliders import ConditionalPG09
    from particle_sliders.pg09_audio import YuE2ARResponseAdapter, AudioResponseCritic, audio_conditions, require_audio_width
    from .backend import YuE2Backend, require_yue2
    from .defaults import MODEL_ID
    from .slider import ParticleSlider
    from .stamp import locked_stamp
    from .train import prepare_bank, _frozen_digest

    # Read the actual local model configuration before allocating its tensors.
    if runtime is None and not options.get('dummy'):
        require_yue2()
        from yue2.storage import resolve_model
        resolved = resolve_model(MODEL_ID, revision=options.get('revision'),
                                 cache_dir=options.get('cache_dir'), local_files_only=True)
        require_audio_width(json.loads((Path(resolved) / 'config.json').read_text())['hidden_size'], **_domain(options))
    elif runtime is not None:
        require_audio_width(runtime.model.config.hidden_size, **_domain(options))

    seed = options.get('seed', 7)
    torch.manual_seed(seed)
    if runtime is None:
        runtime = YuE2Backend(device=options.get('device', 'cpu'), allow_hub=False,
                              dummy=options.get('dummy', False), revision=options.get('revision'),
                              cache_dir=options.get('cache_dir'))
    if bool(runtime.dummy) != options.get('dummy', False):
        raise ValueError('staged runtime dummy provenance differs')
    if any(p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('staged YuE2 base must already be frozen')
    expected_device = torch.device(options.get('device', 'cpu'))
    expected_dtype = torch.float32 if expected_device.type == 'cpu' else torch.bfloat16
    if any(p.device != expected_device or p.dtype != expected_dtype for p in runtime.model.parameters()):
        raise ValueError('staged base device/precision differs')
    restore = _restore_model(runtime.model)
    try:
        # Preserve the existing bridge law and real shared particle ownership.
        rows = _rows(options)
        stamp, _ = locked_stamp(dict(adv_batch=options.get('batch', 2),
            sample_seeds=options.get('sample_seeds', 1), history_tokens=options.get('history_tokens', 4),
            seedbank_sources=len(rows) * options.get('sample_seeds', 1)))
        frozen = _frozen_digest(runtime.model)
        network = ParticleSlider(runtime.model, stamp)
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))}
        adapter = YuE2ARResponseAdapter(runtime, network, model_identity=dict(
            base=runtime.identity, frozen_base_sha256=frozen, product_sources=sources,
            integration_dependencies=dependencies, rows=rows, seed=seed,
            modality='audio', response_kind='ar_last_hidden', waveform_render=False),
            checkpoint_modules=[] if runtime.dummy else list(runtime.model.model.layers), **_domain(options))
        prepared = prepare_bank(runtime, adapter, rows, sample_seeds=options.get('sample_seeds', 1),
                                history_tokens=options.get('history_tokens', 4), seed=seed)
        conditions = audio_conditions(prepared)
        critic = AudioResponseCritic(runtime.model.config.hidden_size, conditions.shape[1]).to(device=expected_device, dtype=torch.float32)
        game = ConditionalPG09(adapter, prepared, critic, conditions=conditions,
            particles=adapter.particles, response_domain='continuous_vector',
            preset=options.get('preset', 'bispf_optimized'), batch_size=options.get('batch', 2),
            seed=seed, total_steps=options.get('steps', 4), sampling=options.get('sampling', 'balanced'),
            **_domain(options))
        return Binding(game, restore, render=None)
    except BaseException:
        restore()
        raise


def _capabilities(options):
    return Capabilities('routed_particle_gan', render=False,
        base_precisions=('float32',) if options.get('device', 'cpu') == 'cpu' else ('bfloat16',),
        render_outputs=(), qualification='YuE2 audio AR last-hidden response only; waveform/image rendering disabled; this draft is untested')


BACKEND = Backend('yue2', _capabilities({}), prepare, validate, _capabilities)
