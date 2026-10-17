"""Explicit reviewed PG profile, staged separately from installed release bytes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_packaged_audio_full_width_explicit_profile():
    profile = os.environ.get('YUE2_PG091_CONTINUOUS')
    if not profile:
        pytest.skip('Set YUE2_PG091_CONTINUOUS to the separately verified PG0.9.1 continuous profile')
    environment = dict(os.environ, PYTHONPATH=profile, CUDA_VISIBLE_DEVICES='',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in ('YUE2_AUDIO_SOURCE', 'YUE2_CORE_SOURCE'):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures/wide_pg091.py')],
                            env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['gate']['resolved_penalty_domain'] == 'continuous'
    assert len(receipt['cases']) == 2
    assert all(c['resume_exact'] and c['export_reload_exact'] for c in receipt['cases'])
    assert receipt['cuda_initialized'] is False
