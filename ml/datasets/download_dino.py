"""Download a small public ONNX visual backbone only; never reads user data."""
import hashlib
import json
import ssl
from pathlib import Path
from urllib.request import urlopen


def main():
    repo = 'Xenova/dinov2-small'
    base = 'facebook/dinov2-small'
    target = Path(__file__).resolve().parents[2]/'artifacts/models/dinov2-small-onnx'
    target.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    def get(url):
        return urlopen(url, context=context, timeout=60)
    with get('https://huggingface.co/api/models/'+repo+'?blobs=true') as response:
        info = json.load(response)
    with get('https://huggingface.co/api/models/'+base+'?blobs=true') as response:
        base_info = json.load(response)
    if base_info.get('cardData', {}).get('license') != 'apache-2.0':
        raise ValueError('Expected Apache-2.0 base weights')
    (target/'repository-metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    wanted = ['onnx/model.onnx', 'config.json', 'preprocessor_config.json', 'README.md']
    for name in wanted:
        entry = next(row for row in info['siblings'] if row['rfilename'] == name)
        output = target/Path(name).name
        expected = (entry.get('lfs') or {}).get('sha256')
        if output.exists() and output.stat().st_size == entry['size']:
            if not expected or hashlib.sha256(output.read_bytes()).hexdigest() == expected:
                continue
        partial = output.with_suffix(output.suffix+'.partial')
        with get('https://huggingface.co/'+repo+'/resolve/'+info['sha']+'/'+name) as response:
            with partial.open('wb') as file:
                for chunk in iter(lambda: response.read(1024*1024), b''):
                    file.write(chunk)
        if partial.stat().st_size != entry['size']:
            raise ValueError('Download size mismatch')
        if expected and hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
            raise ValueError('Weight checksum mismatch')
        partial.replace(output)
        print(json.dumps({'downloaded': name, 'bytes': output.stat().st_size}), flush=True)
    (target/'provenance.json').write_text(json.dumps({
        'repo': repo, 'revision': info['sha'], 'base_model': base,
        'base_revision': base_info['sha'], 'base_license': 'Apache-2.0',
        'conversion': 'ONNX export linked to base model by publisher model card',
        'source': 'https://huggingface.co/'+repo,
        'license_source': 'https://huggingface.co/'+base}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
