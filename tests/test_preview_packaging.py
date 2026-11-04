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
    text = BUILDER.frozen_metadata_template()
    changed = text.replace(anchor, '', 1) if mutation == 'missing' else text + '\n' + anchor
    with pytest.raises(RuntimeError, match='exactly once'):
        BUILDER.preview_metadata(changed)


def test_preview_builder_retains_source_metadata_and_is_deterministic():
    text = BUILDER.frozen_metadata_template()
    result = BUILDER.preview_metadata(text)
    assert result == BUILDER.preview_metadata(text)
    assert 'yue2-hndl = "yue2_sliders.hndl_response:BACKEND"' in result
    assert 'yue2-hndl = "yue2_sliders.hndl_response:INFERENCE"' in result
    assert '"hndl-runtime.lock.json", "hndl_audio_critic.hndl"' in result
    assert BUILDER.frozen_metadata_template() == text
    import hashlib
    assert hashlib.sha256(result.encode()).hexdigest() == '6b8bce5c2675a7fc35db4412920eeb11d9842c4c3a468622767900ea6dfbcea1'


def test_preview_stages_original_runtime_bytes_without_continuous_helpers(tmp_path):
    import hashlib, json
    (tmp_path / 'yue2_sliders').mkdir()
    for name in ('native_assets.py', 'native-assets.lock.json'):
        (tmp_path / 'yue2_sliders' / name).write_text('must not enter historical preview')
    BUILDER.restore_pre_continuous_runtime(ROOT, tmp_path)
    manifest=json.loads((ROOT / 'release_tools/archives/yue2-before-continuous/manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256((tmp_path/relative).read_bytes()).hexdigest() == expected
    assert not (tmp_path/'yue2_sliders/native_assets.py').exists()
    assert not (tmp_path/'yue2_sliders/native-assets.lock.json').exists()
