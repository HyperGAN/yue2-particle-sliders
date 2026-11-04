"""Reviewed response-line locks and native ownership helpers; no training imports."""
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

from particle_sliders.response import ResponseCapabilities

CORE_PIN = '5d68473a098801ee1ab3ebbb12463daef51c6346'
PG_VERSION = '0.9.4'
PROFILE = 'release-0.9.4'
DOMAIN = 'auto'
NATIVE_PROFILE = 'source-801ae1e-continuous-v1'
LOCK = Path(__file__).with_name('lifecycle-runtime.lock.json')
PROTOCOL = 'yue2-native-final-ar-hidden-response-v1'


def engine_options(options):
    from particle_sliders.pg094 import engine_options as shared_options
    return shared_options(options)


def capabilities(options):
    return ResponseCapabilities(generator_arm='routed_particle_gan',
        modality='audio_ar_hidden', response_kind='native_final_ar_hidden',
        response_domain='continuous_vector', response_shape=(1, 32 if options.get('dummy', False) else 2048),
        pg_version=PG_VERSION, **engine_options(options),
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
    engine_options(options)
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


def verify_runtime(options=None):
    """Verify the complete core closure and explicitly selected PG source/domain."""
    selected = engine_options(options or {})
    profile = selected['pg_source_profile']
    lock = json.loads(LOCK.read_text())
    core, pg = lock['core'], lock['particlegan']
    if (lock['schema'], core['commit'], core['version'], pg['version']) != (
            'yue2-response-runtime-v3', CORE_PIN, '0.12.0', PG_VERSION):
        raise ValueError('response code and canonical source lock disagree')
    distribution = metadata.distribution(core['distribution'])
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    archive_sha = direct.get('archive_info', {}).get('hashes', {}).get('sha256')
    vcs = direct.get('vcs_info', {})
    if distribution.version != core['version']:
        raise ValueError('core version differs from the current runtime lock')
    if 'archive_info' in direct:
        if archive_sha != core['wheel_sha256']:
            raise ValueError('core wheel SHA256 differs from the exact current artifact')
    elif (vcs.get('vcs'), vcs.get('commit_id'), direct.get('url', '').removesuffix('.git')) != (
            'git', CORE_PIN, 'https://github.com/HyperGAN/particle-sliders'):
        raise ValueError('core requires exact wheel archive or Git commit provenance')
    if 'particlegan==' + PG_VERSION not in {requirement.replace(' ', '') for requirement in distribution.requires or ()}:
        raise ValueError('core metadata must retain the exact PG0.9.4 dependency')
    pg_distribution = metadata.distribution('particlegan')
    pg_direct = json.loads(pg_distribution.read_text('direct_url.json') or '{}')
    if pg_distribution.version != PG_VERSION or ('archive_info' in pg_direct and
            pg_direct['archive_info'].get('hashes', {}).get('sha256') != pg['profiles'][profile]['wheel_sha256']):
        raise ValueError('ParticleGAN version/archive differs from the selected source profile')
    root = Path(distribution.locate_file('')).resolve()
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for package in ('particle_sliders', 'concept_slider_core')
              for p in sorted((root / package).rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != lock['source_sha256']:
        raise ValueError('installed shared source closure differs from the canonical map')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('particle_sliders', 'concept_slider_core'):
            source = getattr(module, '__file__', None)
            if source is None or not Path(source).resolve().is_relative_to(root) or str(Path(source).resolve().relative_to(root)) not in actual:
                raise ValueError('unqualified shared import origin: ' + name)
    from particle_sliders.pg09 import _released_sources
    if _released_sources(profile) != pg['profiles'][profile]['source_sha256']:
        raise ValueError('PG source digest differs from the canonical lock')
    from particle_sliders.pg094 import validate_pg09_domain
    validate_pg09_domain((1, 1), response_domain='continuous_vector', **selected)
    return dict(core_commit=CORE_PIN, pg_version=PG_VERSION, **selected,
                canonical_sources_sha256=hashlib.sha256(json.dumps(lock['source_sha256'], sort_keys=True).encode()).hexdigest())


def native_sources():
    """Bind product source bytes without importing training modules or data."""
    root = Path(__file__).parent
    names = ('__init__.py', 'response_runtime.py', 'response_native.py', 'backend.py', 'slider.py',
             'stamp.py', 'defaults.py', 'prompts.py', 'lifecycle-runtime.lock.json',
             'native_assets.py', 'native-assets.lock.json', 'lifecycle.py',
             'response_provider.py', 'train.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def frozen_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()



@contextmanager
def observation(runtime, *, native_options=None, expected_numerics=None):
    """Restore native buffer registrations/values, modes and owned hook flags.

    Base weights are immutable and not copied. YuE2 hidden() uses use_cache=False;
    no external KV/token sampler is invoked by the inference response callback.
    """
    import torch
    if native_options is not None:
        from .native_assets import numerical_profile
        if numerical_profile(native_options, runtime.model) != expected_numerics:
            raise ValueError('native YuE2 current numerical identity differs before observation')
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
