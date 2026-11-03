"""Download the pinned public RL dataset, retaining upstream checksums."""
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'sources' / 'Search-VL-RL-8K'
RAW.mkdir(parents=True, exist_ok=True)
metadata_path = RAW / 'upstream_metadata.json'
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
else:
    with urllib.request.urlopen('https://huggingface.co/api/datasets/OpenSearch-VL/Search-VL-RL-8K?blobs=true', timeout=60) as response:
        metadata = json.load(response)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
revision = metadata['sha']
for name in ['README.md', 'rl_data.jsonl', 'images.zip']:
    entry = next(e for e in metadata['siblings'] if e['rfilename'] == name)
    dest = RAW / name
    if not dest.exists() or dest.stat().st_size != entry['size']:
        partial = dest.with_name(dest.name + '.part')
        url = f'https://huggingface.co/datasets/OpenSearch-VL/Search-VL-RL-8K/resolve/{revision}/{name}'
        print(f'Downloading {name}: {entry["size"]:,} bytes; revision {revision}', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--retry', '8', '--retry-delay', '5', '--connect-timeout', '30', '--speed-time', '120', '--speed-limit', '1024', '--continue-at', '-', '--output', str(partial), url], check=True)
        if partial.stat().st_size != entry['size']:
            raise ValueError(f'Size mismatch: {name}')
        partial.rename(dest)
    digest = hashlib.sha256()
    with dest.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = entry.get('lfs', {}).get('sha256')
    if expected and actual != expected:
        raise ValueError(f'SHA256 mismatch: {name}')
    if not expected:
        git_digest = hashlib.sha1(b'blob ' + str(dest.stat().st_size).encode() + b'\0' + dest.read_bytes()).hexdigest()
        if git_digest != entry['blobId']:
            raise ValueError(f'Git blob mismatch: {name}')
    (RAW / (name + '.sha256')).write_text(actual + '  ' + name + '\n')
    print(f'Verified {name}: {actual}', flush=True)
print('All source downloads verified.', flush=True)
