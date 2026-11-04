"""Regenerate the active lifecycle lock from the reviewed PG094 core wheel.

Historical CDF and PG091 locks are never rewritten by this command.
"""
import json
from pathlib import Path
import sys
import zipfile

from lock_hndl_runtime import CORE_COMMIT, CORE_WHEEL_SHA256, ROOT, generate


def lifecycle_lock(wheel):
    reviewed = generate(wheel)
    with zipfile.ZipFile(wheel) as archive:
        release = json.loads(archive.read('particle_sliders/pg094_release.json'))
    return dict(schema='yue2-response-runtime-v2',
        status='CURRENT_SOURCE_COHORT_NATIVE_WIDE_AND_ACOUSTIC_UNQUALIFIED',
        core=dict(distribution='particle-sliders-core', version='0.12.0',
            repository='https://github.com/HyperGAN/particle-sliders',
            commit=CORE_COMMIT, wheel_sha256=CORE_WHEEL_SHA256),
        particlegan=dict(version='0.9.4', release_commit=release['release_commit'],
            release_wheel_sha256=release['wheel_sha256'],
            qualified_source_profile=release['source_profile'],
            qualified_release_sources_sha256=reviewed['pg_sources_sha256']),
        source_sha256=reviewed['source_sha256'],
        qualification=dict(scope='CPU numerical response conformance only; no pretrained/GPU/audio quality claim',
            new_cohort_required=True,
            packet_identity_policy='Old PG091 and CDF packets retain their original source/cohort; no implicit rebinding'))


if __name__ == '__main__':
    destination = ROOT / 'yue2_sliders/lifecycle-runtime.lock.json'
    destination.write_text(json.dumps(lifecycle_lock(Path(sys.argv[1])), indent=2) + '\n')
