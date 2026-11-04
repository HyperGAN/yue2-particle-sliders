"""Regenerate only the active v3 lifecycle lock; preserve archived cohorts."""
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CORE_COMMIT = '5d68473a098801ee1ab3ebbb12463daef51c6346'
CORE_WHEEL_SHA256 = '434aa23289e42e303d45b12c1652d48a5c1c83982773f163b7428e80c124eb11'


def lifecycle_lock(wheel):
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != CORE_WHEEL_SHA256:
        raise ValueError('requires the exact reviewed 5d68473 core wheel')
    with zipfile.ZipFile(wheel) as archive:
        manifest = BytesParser().parsebytes(archive.read('particle_sliders_core-0.12.0.dist-info/METADATA'))
        if manifest['Version'] != '0.12.0' or 'particlegan==0.9.4' not in {
                r.replace(' ', '') for r in manifest.get_all('Requires-Dist', [])}:
            raise ValueError('core version/PG metadata differs')
        sources = {name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
            if name.split('/')[0] in ('particle_sliders', 'concept_slider_core') and not name.endswith('/')}
        profiles = {}
        for filename in ('pg094_release.json', 'pg094_continuous.json'):
            item = json.loads(archive.read('particle_sliders/' + filename))
            profiles[item['source_profile']] = dict(version=item['version'],
                commit=item.get('source_commit', item.get('release_commit')),
                wheel_sha256=item['wheel_sha256'], source_sha256=hashlib.sha256(json.dumps(
                    item['python_sources'], sort_keys=True, allow_nan=False).encode()).hexdigest())
    return dict(schema='yue2-response-runtime-v3',
        status='EXPLICIT_CONTINUOUS_SOURCE_CANDIDATE_NATIVE_GPU_AND_ACOUSTIC_UNQUALIFIED',
        core=dict(distribution='particle-sliders-core', version='0.12.0',
            repository='https://github.com/HyperGAN/particle-sliders',
            commit=CORE_COMMIT, wheel_sha256=CORE_WHEEL_SHA256),
        particlegan=dict(version='0.9.4', release_commit=profiles['release-0.9.4']['commit'], profiles=profiles),
        source_sha256=sources, qualification=dict(
            scope='CPU numerical response conformance and explicit native source candidate; no pretrained GPU/audio quality claim',
            packet_identity_policy='Archived PG091/CDF/HNDL packets and source locks remain unchanged; no implicit rebinding'))


if __name__ == '__main__':
    destination = ROOT / 'yue2_sliders/lifecycle-runtime.lock.json'
    destination.write_text(json.dumps(lifecycle_lock(Path(sys.argv[1])), indent=2) + '\n')
