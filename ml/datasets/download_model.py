"""Download public weights only; never reads or uploads videos."""
import argparse
import fnmatch
import hashlib
import json
import os
import ssl
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['2b', '4b'], default='2b')
    args = parser.parse_args()
    size = args.variant.upper()
    repo = f'mlx-community/Qwen3-VL-{size}-Instruct-4bit'
    root = Path(__file__).resolve().parents[2]
    target = root / f'artifacts/models/qwen3-vl-{args.variant}-4bit'
    target.mkdir(parents=True, exist_ok=True)
    # Standard-library downloader avoids blocking on ML dependencies.
    context = ssl.create_default_context()
    def fetch(url, headers=None):
        return urlopen(Request(url, headers=headers or {}), context=context, timeout=60)
    with fetch('https://huggingface.co/api/models/' + repo + '?blobs=true') as response:
        info = json.load(response)
    revision = info['sha']
    if (info.get('cardData') or {}).get('license') != 'apache-2.0':
        raise ValueError('Expected Apache-2.0 license; review repository metadata')
    (target / 'repository-metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    patterns = ['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'LICENSE*', 'README.md']
    for entry in info['siblings']:
        name = entry['rfilename']
        if '/' in name or not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        final = target / name
        expected_size = entry.get('size')
        expected_sha = (entry.get('lfs') or {}).get('sha256')
        if final.exists() and (expected_size is None or final.stat().st_size == expected_size):
            if not expected_sha or hashlib.sha256(final.read_bytes()).hexdigest() == expected_sha:
                continue
        partial = target / (name + '.partial')
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                url = 'https://huggingface.co/' + repo + '/resolve/' + revision + '/' + quote(name)
                with fetch(url, {'Range': f'bytes={offset}-'} if offset else {}) as response:
                    append = offset > 0 and response.status == 206
                    with partial.open('ab' if append else 'wb') as file:
                        for chunk in iter(lambda: response.read(1024 * 1024), b''):
                            file.write(chunk)
                if expected_size is not None and partial.stat().st_size != expected_size:
                    raise RuntimeError('Incomplete model file: ' + name)
                if expected_sha and hashlib.sha256(partial.read_bytes()).hexdigest() != expected_sha:
                    raise RuntimeError('Model checksum mismatch: ' + name)
                partial.replace(final)
                print('Downloaded:', name, final.stat().st_size, 'bytes', flush=True)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    provenance = {'repo': repo, 'revision': revision, 'license': 'Apache-2.0',
                  'source': 'https://huggingface.co/' + repo,
                  'base_model': f'Qwen/Qwen3-VL-{size}-Instruct'}
    (target / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Model weights saved locally:', target)


if __name__ == '__main__':
    main()
