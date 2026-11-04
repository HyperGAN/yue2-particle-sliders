"""Every preview metadata mutation must fail closed on missing/duplicate anchors."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('preview_builder', ROOT / 'release_tools/build_hndl_preview.py')
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
ANCHORS = [
    'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@ddb41423bcefc1fe76cd94d0f9b367353b3ea137#subdirectory=packages/particle-sliders-core',
    'version = "0.3.0"',
    'requires-python = ">=3.11"',
    'yue2_sliders = ["lifecycle-runtime.lock.json"]',
    'yue2 = "yue2_sliders.lifecycle:BACKEND"',
    'yue2 = "yue2_sliders.response_native:INFERENCE"',
]


@pytest.mark.parametrize('anchor', ANCHORS)
@pytest.mark.parametrize('mutation', ['missing', 'duplicate'])
def test_preview_builder_rejects_every_ambiguous_substitution(anchor, mutation):
    text = (ROOT / 'pyproject.toml').read_text()
    changed = text.replace(anchor, '', 1) if mutation == 'missing' else text + '\n' + anchor
    with pytest.raises(RuntimeError, match='exactly once'):
        BUILDER.preview_metadata(changed)


def test_preview_builder_retains_source_metadata_and_is_deterministic():
    text = (ROOT / 'pyproject.toml').read_text()
    result = BUILDER.preview_metadata(text)
    assert result == BUILDER.preview_metadata(text)
    assert 'yue2-hndl = "yue2_sliders.hndl_response:BACKEND"' in result
    assert 'yue2-hndl = "yue2_sliders.hndl_response:INFERENCE"' in result
    assert '"hndl-runtime.lock.json", "hndl_audio_critic.hndl"' in result
    assert (ROOT / 'pyproject.toml').read_text() == text
