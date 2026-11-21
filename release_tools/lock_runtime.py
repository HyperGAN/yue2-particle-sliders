"""Regenerate the current YuE2 lock from the reviewed installed Git wheel."""
from importlib.metadata import distribution
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CORE_COMMIT = 'ec1161f5d74481be3540cb0dfa63d0e8f8b54e90'


def lock_runtime(destination):
    installed = distribution('particle-sliders-core')
    origin = json.loads(installed.read_text('direct_url.json') or '{}')
    if (installed.version != '0.15.2'
            or origin.get('vcs_info', {}).get('commit_id') != CORE_COMMIT
            or origin.get('url') != 'https://github.com/HyperGAN/particle-sliders.git'
            or origin.get('subdirectory') != 'packages/particle-sliders-core'):
        raise ValueError('install the exact reviewed Particle Sliders core source')
    release = json.loads(installed.locate_file('particle_sliders/engine_release.json').read_text())
    if release['version'] != '0.11.1':
        raise ValueError('core must use the published ParticleGAN 0.11.1 release')
    files = (str(path) for path in installed.files or ())
    sources = {name: hashlib.sha256(installed.locate_file(name).read_bytes()).hexdigest()
               for name in files
               if name.startswith(('particle_sliders/', 'concept_slider_core/'))
               and '__pycache__' not in Path(name).parts and not name.endswith('.pyc')}
    value = json.loads(destination.read_text())
    value['source_sha256'] = sources
    value['particlegan'] = dict(
        version=release['version'], release_commit=release['source_commit'],
        profiles={release['source_profile']: dict(
            version=release['version'], commit=release['source_commit'],
            wheel_sha256=release['wheel_sha256'],
            source_sha256=hashlib.sha256(json.dumps(
                release['python_sources'], sort_keys=True, allow_nan=False).encode()).hexdigest())})
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    lock_runtime(ROOT / 'yue2_sliders/lifecycle-runtime.lock.json')
