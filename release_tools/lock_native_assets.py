"""Audit existing exact model/upstream files into the separate native asset lock."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('yue2_native_assets', ROOT / 'yue2_sliders/native_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def generate(model, upstream):
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != assets.UPSTREAM_COMMIT:
        raise ValueError('requires the exact pinned upstream source checkout')
    config = json.loads((model / 'config.json').read_text())
    if {key: config.get(key) for key in assets.ARCHITECTURE} != assets.ARCHITECTURE:
        raise ValueError('cached model architecture differs')
    return dict(schema='yue2-native-assets-v1', model_id=assets.MODEL_ID,
        revision=assets.MODEL_REVISION, upstream_commit=assets.UPSTREAM_COMMIT,
        architecture=assets.ARCHITECTURE, runtime_versions=assets.VERSIONS,
        model_sources_sha256=assets.sources(model),
        upstream_sources_sha256=assets.sources(upstream / 'src/yue2'),
        qualification='CPU verified immutable assets/source candidate; no pretrained GPU or acoustic quality claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--upstream', required=True, type=Path)
    args = parser.parse_args()
    (ROOT / 'yue2_sliders/native-assets.lock.json').write_text(json.dumps(
        generate(args.model, args.upstream), indent=2) + '\n')
