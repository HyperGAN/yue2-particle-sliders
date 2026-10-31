"""Regenerate only the active modern source/data lock from its reviewed core wheel."""
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import sys
import zipfile

CORE_COMMIT = 'a794e3b4a8813b012f2c4faa3da82b78d4a90214'
CORE_WHEEL_SHA256 = '22e47ced4af8cb874d818db2f4006738f5f978cfcff324693931b6ab23f21c8a'
ROOT = Path(__file__).resolve().parents[1]


def generate(wheel):
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != CORE_WHEEL_SHA256:
        raise ValueError('requires the reviewed a794 core wheel')
    with zipfile.ZipFile(wheel) as archive:
        manifest = BytesParser().parsebytes(archive.read('particle_sliders_core-0.12.0.dist-info/METADATA'))
        if manifest['Version'] != '0.12.0' or 'particlegan==0.9.4' not in {
                r.replace(' ', '') for r in manifest.get_all('Requires-Dist', [])}:
            raise ValueError('core version/PG metadata differs')
        names = sorted(name for name in archive.namelist()
            if name.split('/')[0] in ('particle_sliders', 'concept_slider_core') and not name.endswith('/'))
        sources = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}
        release = json.loads(archive.read('particle_sliders/pg094_release.json'))
        if (release['version'], release['source_profile']) != ('0.9.4', 'release-0.9.4'):
            raise ValueError('unqualified PG profile')
    return dict(schema='yue2-hypergan-hndl-runtime-v2', core_commit=CORE_COMMIT,
        core_version='0.12.0', core_wheel_sha256=CORE_WHEEL_SHA256,
        pg_version='0.9.4', pg_sources_sha256=hashlib.sha256(json.dumps(
            release['python_sources'], sort_keys=True, allow_nan=False).encode()).hexdigest(),
        source_sha256=sources, status='PUBLIC_SOURCE_CANDIDATE_NOT_A_PACKAGE_RELEASE',
        historical_packet_policy='Old preview packets retain their original source/cohort; no implicit rebinding')


if __name__ == '__main__':
    destination = ROOT / 'cohorts/hndl/yue2_sliders/hndl-runtime.lock.json'
    destination.write_text(json.dumps(generate(Path(sys.argv[1])), indent=2) + '\n')
