"""Run the unchanged upstream agent while recording transport status, never credentials."""
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(os.environ.get('EVAL_OFFICIAL_ROOT', str(Path(__file__).resolve().parents[2] / 'outputs/official_eval_dependencies')))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
REPO = BASE / 'AgentVista'
sys.path.insert(0, str(REPO))
from engine.api import api_caller

original_request = api_caller._make_api_request
if os.environ.get('EVAL_LOCAL_MODEL'):
    import requests
    from evaluation.local_api import LocalCompletionClient
    local = LocalCompletionClient(os.environ['EVAL_LOCAL_MODEL'],
                                  device=os.environ.get('EVAL_LOCAL_DEVICE', 'mps'), answer_tag='answer')
    def original_request(end_point, headers, payload, api_name='API'):
        body = local.create(**payload).model_dump(exclude_none=True)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(body).encode()
        return response, None
    real_post = requests.post
    def local_model_post(url, *args, **kwargs):
        if url == 'http://local-sft.invalid/v1/chat/completions':
            return original_request(url, {}, kwargs['json'])[0]
        return real_post(url, *args, **kwargs)
    requests.post = local_model_post


def audited_request(*args, **kwargs):
    response, error = original_request(*args, **kwargs)
    record = {'http_status':response.status_code if response is not None else None,
              'request_error':error}
    if response is not None:
        try:
            body = response.json()
            if isinstance(body.get('error'), dict):
                record['error_message'] = str(body['error'].get('message',''))[:600]
                record['error_code'] = body['error'].get('code')
            if body.get('usage'):
                record['usage'] = body['usage']
        except (ValueError, AttributeError):
            record['response_bytes'] = len(response.content)
    key = os.environ.get('REASONING_API_KEY','')
    serialized = json.dumps(record, ensure_ascii=False)
    if key:
        serialized = serialized.replace(key,'[REDACTED]')
    with Path(os.environ['AGENTVISTA_TRANSPORT_AUDIT']).open('a') as stream:
        stream.write(serialized+'\n')
    if record['http_status'] != 200:
        print('[Transport] '+serialized, flush=True)
    return response, error


api_caller._make_api_request = audited_request
sys.argv[0] = str(REPO/'infer.py')
runpy.run_path(sys.argv[0], run_name='__main__')
