"""Prepare the tested Python dependencies and original checkpoint for Xeus mounts."""

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen


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
    sources = json.loads((ROOT / 'content/sources.json').read_text())
    checkpoint = model / 'pytorch_model.bin'
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix('.partial')
        try:
            print('Downloading the original pinned Whisper base checkpoint...', flush=True)
            with urlopen(sources['checkpoint'] + '?download=true', timeout=120) as response:
                with temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
            if sha256(temporary) != sources['checkpoint_sha256']:
                raise RuntimeError('Model checkpoint failed its SHA-256 check')
            temporary.replace(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(checkpoint) != sources['checkpoint_sha256']:
        raise RuntimeError('Model checkpoint failed its SHA-256 check')
    print('Whisper model and Python dependencies verified and ready for Xeus.', flush=True)


if __name__ == '__main__':
    main()
