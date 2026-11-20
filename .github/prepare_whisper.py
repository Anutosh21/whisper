"""Prepare Python dependencies and model configuration for Xeus mounts."""

import hashlib
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / '_build'
DEPENDENCIES_SHA256 = '89fbcc3a52bd78503544845d8d35a478d41a99bce80bab7e413ba46880e785e3'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    bundle = ROOT / 'vendor/whisper-python-dependencies.tar.gz'
    if sha256(bundle) != DEPENDENCIES_SHA256:
        raise RuntimeError('Python dependency bundle failed its SHA-256 check')
    with tarfile.open(bundle) as archive:
        members = [member for member in archive.getmembers()
                   if member.name.startswith('lib/python3.14/site-packages/')]
        archive.extractall(ASSETS / 'dependencies', members=members, filter='data')

    model = ASSETS / 'model'
    shutil.copytree(ROOT / 'model', model, dirs_exist_ok=True)
    # Older builds preloaded the checkpoint; the notebook now fetches it on demand.
    (model / 'pytorch_model.bin').unlink(missing_ok=True)
    print('Whisper configuration and Python dependencies ready for Xeus.', flush=True)


if __name__ == '__main__':
    main()
