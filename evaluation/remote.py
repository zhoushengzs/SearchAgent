"""OpenRouter transport and bounded parallel work; no model weights are loaded."""
from __future__ import annotations

import base64
import copy
import io
import json
import math
import random
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from PIL import Image


@contextmanager
def output_lock(directory):
    """Prevent two processes from billing/writing the same run concurrently."""
    import fcntl
    with (Path(directory) / '.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another process is writing this output directory.') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class RemoteAPIError(RuntimeError):
    pass


class RemoteFatalError(RemoteAPIError):
    """Authentication, credit, or request configuration requires intervention."""


class RemoteResponseError(RemoteAPIError):
    pass


def parallel_results(function, items, workers, stop=None):
    """At most workers jobs in flight; caller alone writes completed records."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        def fill():
            while len(pending) < workers and not (stop and stop.is_set()):
                item = next(iterator, None)
                if item is None:
                    break
                pending.add(pool.submit(function, item))
        fill()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            fill()


class APIClient:
    def __init__(self, key, base_url='https://openrouter.ai/api/v1', attempts=4, timeout=120):
        self.key, self.url = key, base_url.rstrip('/') + '/chat/completions'
        self.attempts, self.timeout = attempts, timeout
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cooldown = 0.0

    def _pause(self, response, attempt):
        delay = min(30, 2 ** attempt) + random.random()
        if response is not None:
            value = response.headers.get('Retry-After')
            if value:
                try:
                    delay = max(delay, float(value))
                except ValueError:
                    try:
                        delay = max(delay, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError):
                        pass
        with self.lock:
            self.cooldown = max(self.cooldown, time.monotonic() + delay)

    def post(self, payload):
        for attempt in range(self.attempts):
            while True:
                if self.stop.is_set():
                    raise RemoteFatalError('Remote requests stopped; check credentials, credits or request settings.')
                with self.lock:
                    remaining = self.cooldown - time.monotonic()
                if remaining <= 0:
                    break
                self.stop.wait(min(remaining, 1))
            response = None
            try:
                response = requests.post(self.url, headers={'Authorization': 'Bearer ' + self.key},
                                         json=payload, timeout=self.timeout)
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                error = body.get('error') if isinstance(body, dict) else None
                status = response.status_code
                if error and isinstance(error, dict):
                    try:
                        status = int(error.get('code', status if status >= 400 else 502))
                    except (TypeError, ValueError):
                        status = 502
                if status in {400, 401, 402, 403, 404, 422}:
                    hint = {400: 'Invalid request parameters.', 401: 'Check the API key.',
                            402: 'Check credits and the key spending limit.',
                            403: 'Check account permissions and provider policy.',
                            404: 'No compatible endpoint; check the model and requested parameters.',
                            422: 'Check the request format.'}[status]
                    print(f'Remote HTTP {status}: {hint}', file=sys.stderr, flush=True)
                    self.stop.set()
                    raise RemoteFatalError(f'Remote HTTP {status}; check request, key, credits and provider availability.')
                if status == 429 or status in {408, 409} or status >= 500 or error:
                    raise RemoteAPIError(f'Remote HTTP {status}')
                if status >= 400 or not isinstance(body, dict) or not body.get('choices'):
                    raise RemoteResponseError('Remote response has no completion.')
                return body
            except (requests.RequestException, RemoteAPIError) as exc:
                if isinstance(exc, RemoteFatalError):
                    raise
                if attempt + 1 == self.attempts:
                    # Never export raw server error bodies or authorization headers.
                    raise RemoteAPIError('Remote request failed after bounded retries.') from None
                self._pause(response, attempt)


def encode_image(path, max_pixels):
    """Bound transmitted pixels, maintaining aspect ratio on Qwen's 32px grid."""
    with Image.open(path) as source:
        image = source.convert('RGB')
        width, height = image.size
        scale = min(1.0, math.sqrt(max_pixels / (width * height)))
        w = max(32, math.floor(width * scale / 32) * 32)
        h = max(32, math.floor(height * scale / 32) * 32)
        while w * h > max_pixels:
            if w >= h and w > 32:
                w -= 32
            elif h > 32:
                h -= 32
            else:
                raise ValueError('Image pixel budget is too small')
        if image.size != (w, h):
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii'), (w, h)


class OpenRouterModel:
    def __init__(self, model, tokenizer, client, provider=None):
        self.model, self.tokenizer, self.client, self.provider = model, tokenizer, client, provider
        self.tokenizer_lock = threading.Lock()

    def prepare(self, messages, tools):
        from evaluation.run import parse_output
        prepared, counting = [], copy.deepcopy(messages)
        image_tokens = 0
        pending_id = None
        for index, message in enumerate(messages):
            role, content = message['role'], message['content']
            if role == 'assistant':
                action, value = parse_output(content)
                if action == 'tool':
                    pending_id = f'call_{index}'
                    # Native API envelopes carry the same validated action. No observer call is added.
                    thought = content.split('<tool_call>', 1)[0].strip()
                    prepared.append({'role': 'assistant', 'content': thought or None, 'tool_calls': [
                        {'id': pending_id, 'type': 'function', 'function': {
                            'name': value['name'], 'arguments': json.dumps(value['arguments'], ensure_ascii=False)}}]})
                    continue
            if role == 'tool':
                if pending_id is None:
                    raise ValueError('Tool result has no matching model action')
                prepared.append({'role': role, 'content': content, 'tool_call_id': pending_id})
                pending_id = None
                continue
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if block['type'] == 'image':
                        encoded, (w, h) = encode_image(block['image'], block['max_pixels'])
                        blocks.append({'type': 'image_url', 'image_url': {'url': encoded}})
                        # Qwen3-VL uses 16px patches merged 2x2. Token count is an estimate
                        # of the remote chat template; retain actual API usage in every turn.
                        image_tokens += max(w * h, 65536) // 1024 - 1
                    else:
                        blocks.append(copy.deepcopy(block))
                prepared.append({'role': role, 'content': blocks})
            else:
                prepared.append({'role': role, 'content': content})
        kwargs = {'tools': tools} if tools else {}
        with self.tokenizer_lock:
            token_ids = self.tokenizer.apply_chat_template(counting, tokenize=True,
                                                           add_generation_prompt=True, return_dict=False, **kwargs)
        return prepared, len(token_ids) + image_tokens

    def generate(self, messages, tools, max_tokens, context_tokens, **generation_kwargs):
        wire, estimated = self.prepare(messages, tools)
        if estimated + max_tokens > context_tokens:
            return {'text': '', 'status': 'context_limit', 'input_tokens': estimated, 'new_tokens': 0,
                    'token_count_source': 'local_qwen_template_estimate'}
        routing = {'require_parameters': True, 'allow_fallbacks': False}
        if self.provider:
            routing['only'] = [self.provider]
        payload = {'model': self.model, 'messages': wire, 'temperature': 0,
                   'max_tokens': max_tokens, 'stream': False, 'provider': routing, 'transforms': []}
        if tools:
            # Some Qwen endpoints support tools but do not advertise
            # parallel_tool_calls. Requiring that parameter excludes them all.
            # The shared prompt requests one action, and the response check below
            # rejects multiple calls before the agent can execute any of them.
            payload.update(tools=tools, tool_choice='auto')
        repetition_penalty = generation_kwargs.get('repetition_penalty')
        if repetition_penalty is not None and float(repetition_penalty) != 1.0:
            payload['repetition_penalty'] = float(repetition_penalty)
        body = self.client.post(payload)
        if body.get('model') != self.model:
            self.client.stop.set()
            raise RemoteFatalError('Returned model differs from the requested base model')
        choice, usage = body['choices'][0], body.get('usage') or {}
        message = choice.get('message') or {}
        text = message.get('content') or ''
        if not isinstance(text, str):
            raise RemoteResponseError('Non-text completion')
        status = 'ok'
        calls = message.get('tool_calls') or []
        if calls:
            if len(calls) != 1 or '<tool_call>' in text or '<response>' in text:
                status = 'invalid_output'
            else:
                try:
                    function = calls[0]['function']
                    arguments = json.loads(function['arguments'])
                    if not isinstance(arguments, dict) or not isinstance(function['name'], str):
                        raise ValueError('Invalid tool arguments')
                    text += '<tool_call>' + json.dumps({'name': function['name'], 'arguments': arguments}, ensure_ascii=False) + '</tool_call>'
                except (ValueError, KeyError, TypeError):
                    status = 'invalid_output'
        finish = choice.get('finish_reason')
        if finish == 'length':
            status = 'length'
        elif finish not in {'stop', 'tool_calls'}:
            raise RemoteResponseError('Unexpected or missing finish reason')
        if not text.strip() and status == 'ok':
            raise RemoteResponseError('Empty completion')
        prompt_tokens = usage.get('prompt_tokens')
        if not isinstance(prompt_tokens, int):
            raise RemoteResponseError('Provider omitted token usage; cannot audit context budget')
        if prompt_tokens + max_tokens > context_tokens:
            status = 'context_limit'
        return {'text': text, 'status': status, 'input_tokens': prompt_tokens,
                'estimated_input_tokens': estimated, 'new_tokens': usage.get('completion_tokens'),
                'usage': usage, 'request_id': body.get('id'), 'returned_model': body.get('model'),
                'provider': body.get('provider'), 'finish_reason': finish,
                'reasoning': message.get('reasoning'), 'native_tool_calls': calls}
