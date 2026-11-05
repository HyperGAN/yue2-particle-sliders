"""Regenerate only the active PG0.9.5 lifecycle lock; preserve archived cohorts."""
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CORE_COMMIT = '3dd672a7f9b8c169daef90fd63e4bb42b9ddc423'
CORE_WHEEL_SHA256 = 'af1f64ee624d103a9f77b949b57a97cfeddf37d0e56a8aa5cc93a7d94855dda2'


def lifecycle_lock(wheel):
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != CORE_WHEEL_SHA256:
        raise ValueError('requires the exact reviewed 3dd672a core wheel')
    with zipfile.ZipFile(wheel) as archive:
        manifest = BytesParser().parsebytes(archive.read('particle_sliders_core-0.13.0.dist-info/METADATA'))
        if manifest['Version'] != '0.13.0' or 'particlegan==0.9.5' not in {
                r.replace(' ', '') for r in manifest.get_all('Requires-Dist', [])}:
            raise ValueError('core version/PG metadata differs')
        sources = {name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
            if name.split('/')[0] in ('particle_sliders', 'concept_slider_core') and not name.endswith('/')}
        profiles = {}
        for filename in ('pg095_release.json',):
            item = json.loads(archive.read('particle_sliders/' + filename))
            profiles[item['source_profile']] = dict(version=item['version'],
                commit=item.get('source_commit', item.get('release_commit')),
                wheel_sha256=item['wheel_sha256'], source_sha256=hashlib.sha256(json.dumps(
                    item['python_sources'], sort_keys=True, allow_nan=False).encode()).hexdigest())
    return dict(schema='yue2-response-runtime-v4-pg095',
        status='PG095_RELEASE_NATIVE_CONTINUOUS_SOURCE_READY_NOT_NATIVE_QUALIFIED',
        core=dict(distribution='particle-sliders-core', version='0.13.0',
            repository='https://github.com/HyperGAN/particle-sliders',
            commit=CORE_COMMIT, wheel_sha256=CORE_WHEEL_SHA256),
        particlegan=dict(version='0.9.5', release_commit=profiles['release-0.9.5']['commit'], profiles=profiles),
        source_sha256=sources, qualification=dict(
            scope='CPU numerical response conformance and explicit native source candidate; no pretrained GPU/audio quality claim',
            packet_identity_policy='Archived PG091/CDF/HNDL packets and source locks remain unchanged; no implicit rebinding'))


if __name__ == '__main__':
    destination = ROOT / 'yue2_sliders/lifecycle-runtime.lock.json'
    destination.write_text(json.dumps(lifecycle_lock(Path(sys.argv[1])), indent=2) + '\n')
