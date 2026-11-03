"""Share real evidence and identical judge requests across experimental conditions."""
import base64
from contextlib import contextmanager
import fcntl
import importlib
import json
import os
from pathlib import Path
import tempfile

from evaluation.run import fingerprint


def memo(directory, namespace, request, loader):
    identity = json.loads(json.dumps({'schema': 'unified6-snapshot-v1', 'request': request}, sort_keys=True))
    key = fingerprint(identity)
    folder = Path(directory)/namespace
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(key+'.json')
    with (folder/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identity'] != identity or saved['sha256'] != fingerprint(saved['value']):
                raise ValueError('Shared snapshot changed; do not silently replace it')
            return saved['value'], True, key
        # Failed requests are recorded in the normal action logs, never fabricated as successes.
        value = loader()
        saved = {'identity': identity, 'value': value, 'sha256': fingerprint(value)}
        fd, temporary = tempfile.mkstemp(prefix=key+'.', dir=folder)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(saved, stream, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return value, False, key


@contextmanager
def evidence_snapshots(directory):
    evidence = importlib.import_module('project_runtime.evidence')
    original_search = evidence.EvidenceTools._search
    original_fetch = evidence.fetch_public
    original_execute = evidence.EvidenceTools.execute

    def search(instance, query, images):
        if instance.kb:
            return original_search(instance, query, images)
        result, _, _ = memo(directory, 'search', {'query': query, 'images': images, 'num': 5},
                            lambda: original_search(instance, query, images))
        for row in result['results']:
            instance.urls.add(row['url'])
            if row.get('image_url'):
                instance.image_urls.add(row['image_url'])
        return result

    def fetch(url, *, max_bytes=8*1024*1024, timeout=30):
        def request():
            raw, final_url, content_type, status = original_fetch(url, max_bytes=max_bytes, timeout=timeout)
            return {'body': base64.b64encode(raw).decode(), 'final_url': final_url,
                    'content_type': content_type, 'status': status}
        value, _, _ = memo(directory, 'public-fetch', {'url': url, 'max_bytes': max_bytes, 'timeout': timeout}, request)
        raw = base64.b64decode(value['body'], validate=True)
        if len(raw) > max_bytes:
            raise ValueError('Saved evidence exceeds the common byte limit')
        return raw, value['final_url'], value['content_type'], value['status']

    def execute(instance, tool, args, step):
        result = original_execute(instance, tool, args, step)
        # Keep actual timestamps in raw artifacts; omit wall-clock metadata from model input.
        result.pop('observed_at', None)
        return result

    evidence.EvidenceTools._search, evidence.fetch_public, evidence.EvidenceTools.execute = search, fetch, execute
    try:
        yield
    finally:
        evidence.EvidenceTools._search, evidence.fetch_public, evidence.EvidenceTools.execute = original_search, original_fetch, original_execute
