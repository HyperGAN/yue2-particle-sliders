"""Read-only installed-wheel audit receipt; no pretrained models or CUDA."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

product_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(product_root))
from yue2.integration import configure_integration
runtime = configure_integration()
from particle_sliders import ConditionalPG09
from particle_sliders.pg09 import _released_sources
from particle_sliders.pg09_audio import AudioResponseCritic, AudioRow, YuE2ARResponseAdapter, audio_conditions
from particle_sliders.pg09_backends import AdapterParameters
import particlegan
import torch

core_distribution = importlib.metadata.distribution('particle-sliders-core')
pg_distribution = importlib.metadata.distribution('particlegan')
source_files = sorted((product_root / 'yue2').glob('*.py')) + sorted((product_root / 'scripts').glob('*.py'))
source_files += [product_root / 'comfy_yue2.py', product_root / '__init__.py']
origins = {value.__name__: inspect.getsourcefile(value) for value in (
    ConditionalPG09, AudioResponseCritic, AudioRow, YuE2ARResponseAdapter, audio_conditions, AdapterParameters)}
assert all(Path(path).resolve().is_relative_to(Path(sys.prefix)) for path in origins.values())
assert not any(name.startswith('conceptmod.textsliders') for name in sys.modules)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(
    python=sys.executable, prefix=sys.prefix, python_version=sys.version,
    torch=str(torch.__version__), cuda_initialized=False,
    dependency=runtime, imports=origins, particlegan_origin=particlegan.__file__,
    core_direct_url=json.loads(core_distribution.read_text('direct_url.json')),
    pg_direct_url=json.loads(pg_distribution.read_text('direct_url.json')),
    pg_release_sources_sha256=_released_sources('release'),
    product_sources={str(p.relative_to(product_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    shared_checkout_paths=[path for path in sys.path if 'particle-sliders-core/src' in path or 'audio-adapters-' in path],
    scope='Installed wheel CPU software qualification; no native pretrained runtime/audio'), indent=2))
