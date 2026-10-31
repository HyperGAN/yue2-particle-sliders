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


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    text = (ROOT / 'pyproject.toml').read_text()
    old = 'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@f6807a91546e43e198645a3f8925d2dc7c9e03b4#subdirectory=packages/particle-sliders-core'
    new = 'particle-sliders-core[hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231#subdirectory=packages/particle-sliders-core'
    if text.count(old) != 1 or text.count('version = "0.2.0"') != 1 or text.count('requires-python = ">=3.10"') != 1:
        raise RuntimeError('default packaging changed; review the preview source cohort before building')
    text = text.replace(old, new).replace('version = "0.2.0"', 'version = "0.3.0.dev0"')
    text = text.replace('requires-python = ">=3.10"', 'requires-python = ">=3.11,<3.15"')
    # Only the new factories are registered in the preview environment.
    text = text.replace('yue2 = "yue2_sliders.lifecycle:BACKEND"\n', '')
    text = text.replace('yue2 = "yue2_sliders.response_native:INFERENCE"\n', '')
    with tempfile.TemporaryDirectory(prefix='yue2-hndl-build-') as staging:
        source = Path(staging)
        for name in ('yue2_sliders', 'configs'):
            shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('LICENSE', 'LIFECYCLE.md'):
            shutil.copyfile(ROOT / name, source / name)
        (source / 'pyproject.toml').write_text(text)
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                        '--no-index', str(source), '-w', str(output)], check=True)


if __name__ == '__main__':
    main()
