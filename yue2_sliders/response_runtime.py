"""Reviewed response-line locks and native ownership helpers; no training imports."""
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

from particle_sliders.response import ResponseCapabilities

CORE_PIN = 'f6807a91546e43e198645a3f8925d2dc7c9e03b4'
PG_VERSION = '0.9.1'
PROFILE = 'release'
DOMAIN = 'auto'
LOCK = Path(__file__).with_name('lifecycle-runtime.lock.json')
PROTOCOL = 'yue2-native-final-ar-hidden-response-v1'


def capabilities(options):
    return ResponseCapabilities(generator_arm='routed_particle_gan',
        modality='audio_ar_hidden', response_kind='native_final_ar_hidden',
        response_domain='continuous_vector', response_shape=(1, 32 if options.get('dummy', False) else 2048),
        pg_version=PG_VERSION, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN,
        base_precisions=(options.get('base_precision', 'float32' if options.get('device', 'cpu') == 'cpu' else 'bfloat16'),),
        unavailable_reasons=(('render', 'YuE2 numerical AR hidden responses are not tokens or waveforms; acoustic rendering is unqualified'),),
        qualification='Product source draft only; tiny/native/GPU/acoustic execution unqualified')


def validate_native_options(options):
    if type(options.get('dummy', False)) is not bool:
        raise ValueError('dummy must be a boolean')
    device = options.get('device', 'cpu')
    if not isinstance(device, str) or not (device == 'cpu' or re.fullmatch(r'cuda:[0-9]+', device)):
        raise ValueError('device must be cpu or an explicitly admitted cuda device')
    precision = 'float32' if device == 'cpu' else 'bfloat16'
    if options.get('dummy') and device != 'cpu':
        raise ValueError('dummy is CPU only')
    if options.get('base_precision', precision) != precision:
        raise ValueError('requested base precision differs from the native loader')
    seed = options.get('base_seed', 7)
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError('base_seed must be a bounded nonnegative integer')
    if (options.get('pg_version', PG_VERSION), options.get('pg_source_profile', PROFILE),
            options.get('penalty_input_domain', DOMAIN)) != (PG_VERSION, PROFILE, DOMAIN):
        raise ValueError('this proposal admits only reviewed PG0.9.1 release/auto; the old continuous profile is unrecovered')
    for key in ('revision', 'cache_dir'):
        if key in options and options[key] is not None and not isinstance(options[key], str):
            raise ValueError(key + ' must be a local loader string or null')


def validate_staged(runtime, options):
    """Inspect borrowed runtime metadata without touching devices or RNG."""
    if runtime.model.config.hidden_size != capabilities(options).response_shape[1]:
        raise ValueError('staged native width differs from the response descriptor')
    if bool(runtime.dummy) != options.get('dummy', False):
        raise ValueError('staged dummy provenance differs')
    device = options.get('device', 'cpu')
    dtype = 'torch.float32' if device == 'cpu' else 'torch.bfloat16'
    if any(p.requires_grad or str(p.device) != device or str(p.dtype) != dtype for p in runtime.model.parameters()):
        raise ValueError('base must be frozen at the declared device and precision')


def verify_runtime():
    """Use the canonical full source map verbatim; never invent a closure map."""
    lock = json.loads(LOCK.read_text())
    core, pg = lock['core'], lock['particlegan']
    if (core['commit'], core['version'], pg['version'], pg['qualified_source_profile']) != (CORE_PIN, '0.8.2', PG_VERSION, PROFILE):
        raise ValueError('response code and canonical source lock disagree')
    for name, version, sha in [(core['distribution'], core['version'], core['wheel_sha256']),
                               ('particlegan', pg['version'], pg['release_wheel_sha256'])]:
        distribution = metadata.distribution(name)
        direct = json.loads(distribution.read_text('direct_url.json') or '{}')
        archive_sha = direct.get('archive_info', {}).get('hashes', {}).get('sha256')
        if distribution.version != version or archive_sha is not None and archive_sha != sha:
            raise ValueError('version/archive identity differs from the locked wheel for ' + name)
    distribution = metadata.distribution(core['distribution'])
    if 'particlegan==' + PG_VERSION not in {requirement.replace(' ', '') for requirement in distribution.requires or ()}:
        raise ValueError('core metadata must retain the exact PG0.9.1 dependency')
    root = Path(distribution.locate_file('')).resolve()
    # The reviewed owner also qualifies isolated wheel extraction (no direct_url).
    # Exact canonical runtime bytes and all released PG sources are still mandatory.
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for package in ('particle_sliders', 'concept_slider_core')
              for p in sorted((root / package).rglob('*'))
              if p.is_file() and (p.suffix in ('.py', '.json') or p.name == 'LICENSE')}
    if actual != lock['source_sha256']:
        raise ValueError('installed shared source closure differs from the canonical map')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('particle_sliders', 'concept_slider_core'):
            source = getattr(module, '__file__', None)
            if source is None or not Path(source).resolve().is_relative_to(root) or str(Path(source).resolve().relative_to(root)) not in actual:
                raise ValueError('unqualified shared import origin: ' + name)
    from particle_sliders.pg09 import _released_sources
    if _released_sources(PROFILE) != pg['qualified_release_sources_sha256']:
        raise ValueError('PG source digest differs from the canonical lock')
    return dict(core_commit=CORE_PIN, pg_version=PG_VERSION, pg_source_profile=PROFILE,
                canonical_sources_sha256=hashlib.sha256(json.dumps(lock['source_sha256'], sort_keys=True).encode()).hexdigest())


def native_sources():
    """Bind native-only product helpers without training modules or data."""
    root = Path(__file__).parent
    names = ('__init__.py', 'response_runtime.py', 'response_native.py', 'backend.py', 'slider.py',
             'stamp.py', 'defaults.py', 'prompts.py', 'lifecycle-runtime.lock.json')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def frozen_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()



@contextmanager
def observation(runtime):
    """Restore native buffer registrations/values, modes and owned hook flags.

    Base weights are immutable and not copied. YuE2 hidden() uses use_cache=False;
    no external KV/token sampler is invoked by the inference response callback.
    """
    import torch
    from particle_sliders.lifecycle import native_cleanup
    restore = native_cleanup(runtime.model)
    buffers = [(m, dict(m._buffers), set(m._non_persistent_buffers_set)) for m in runtime.model.modules()]
    values = {id(b): (b, b.detach().clone()) for _, registered, _ in buffers for b in registered.values() if b is not None}
    try:
        runtime.model.eval()
        yield
    finally:
        try:
            with torch.no_grad():
                for buffer, value in values.values():
                    if (buffer.shape, buffer.dtype, buffer.device) == (value.shape, value.dtype, value.device):
                        buffer.copy_(value)
                    else:
                        buffer.data = value.clone()
                for module, registered, nonpersistent in buffers:
                    module._buffers.clear()
                    module._buffers.update(registered)
                    module._non_persistent_buffers_set.clear()
                    module._non_persistent_buffers_set.update(nonpersistent)
        finally:
            restore()
