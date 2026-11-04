"""Build the explicit HNDL preview without changing the supported default pins.

Usage: python release_tools/build_hndl_preview.py /absolute/output/directory
Requires the declared patched setuptools build backend in the build environment.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    """Reject missing or ambiguous packaging anchors before every substitution."""
    if text.count(old) != 1:
        raise RuntimeError('preview packaging anchor must occur exactly once: ' + old)
    return text.replace(old, new, 1)


def preview_metadata(text):
    old = 'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@ddb41423bcefc1fe76cd94d0f9b367353b3ea137#subdirectory=packages/particle-sliders-core'
    new = 'particle-sliders-core[hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@ddb41423bcefc1fe76cd94d0f9b367353b3ea137#subdirectory=packages/particle-sliders-core'
    text = replace_once(text, old, new)
    text = replace_once(text, 'version = "0.3.0"', 'version = "0.4.0.dev0"')
    text = replace_once(text, 'requires-python = ">=3.11"', 'requires-python = ">=3.11,<3.15"')
    text = replace_once(text, 'yue2_sliders = ["lifecycle-runtime.lock.json"]',
        'yue2_sliders = ["lifecycle-runtime.lock.json", "hndl-runtime.lock.json", "hndl_audio_critic.hndl"]')
    text = replace_once(text, 'yue2 = "yue2_sliders.lifecycle:BACKEND"',
        'yue2-hndl = "yue2_sliders.hndl_response:BACKEND"')
    return replace_once(text, 'yue2 = "yue2_sliders.response_native:INFERENCE"',
        'yue2-hndl = "yue2_sliders.hndl_response:INFERENCE"')


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    text = preview_metadata((ROOT / 'pyproject.toml').read_text())
    with tempfile.TemporaryDirectory(prefix='yue2-hndl-build-') as staging:
        source = Path(staging)
        for name in ('yue2_sliders', 'configs'):
            shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('hndl_response.py', 'hndl-runtime.lock.json', 'hndl_audio_critic.hndl'):
            shutil.copyfile(ROOT / 'cohorts' / 'hndl' / 'yue2_sliders' / name, source / 'yue2_sliders' / name)
        shutil.copyfile(ROOT / 'LICENSE', source / 'LICENSE')
        shutil.copyfile(ROOT / 'docs/hypergan-hndl-cohort.md', source / 'LIFECYCLE.md')
        (source / 'pyproject.toml').write_text(text)
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                        '--no-index', str(source), '-w', str(output)], check=True)
        subprocess.run([sys.executable, '-c',
                        'import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])',
                        str(output)], cwd=source, check=True)


if __name__ == '__main__':
    main()
