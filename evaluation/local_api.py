"""Local Qwen inference behind the OpenAI message shape used by benchmark suites.

No HTTP server and no remote model inference. Only lossless protocol/name aliases
are adapted; unsupported capabilities are never fabricated.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid

from evaluation.execution import final_content


def canonical_call(name, args, names):
    if name in names:
        return name, args
    aliases = {'vs_web_search': 'web_search', 'vs_image_search': 'image_search',
               'vs_visit': 'visit', 'vs_calculate': 'calculate'}
    mapped = aliases.get(name)
    if mapped in names:
        args = dict(args)
        if name == 'vs_image_search':
            args['search_type'] = 'text'
        return mapped, args
    return name, args


class LocalCompletionClient:
    def __init__(self, model_path, device='mps', answer_tag=None, model=None):
        from evaluation.run import LocalModel
        self.model = model or LocalModel(str(Path(model_path).resolve()), device, 'bfloat16')
        self.path = str(Path(model_path).resolve())
        self.answer_tag = answer_tag
        self.chat = self.completions = self
        self.lock = threading.Lock()

    def create(self, *, messages, tools=None, max_tokens=4096, **kwargs):
        from openai.types.chat import ChatCompletion
        from agent.tools import fetch_public
        from evaluation.run import parse_output
        with self.lock, tempfile.TemporaryDirectory(prefix='searchagent-image-') as directory:
            converted = []
            for message in messages:
                role = message['role']
                content = message.get('content') or ''
                if role == 'assistant' and message.get('tool_calls'):
                    content = content if isinstance(content, str) else ''
                    for call in message['tool_calls']:
                        fn = call['function']
                        args = fn['arguments']
                        args = json.loads(args) if isinstance(args, str) else args
                        content += '<tool_call>' + json.dumps({'name': fn['name'], 'arguments': args}) + '</tool_call>'
                if isinstance(content, list):
                    blocks = []
                    for block in content:
                        if block['type'] == 'text':
                            blocks.append(dict(block))
                        elif block['type'] == 'image_url':
                            source = block['image_url']
                            source = source['url'] if isinstance(source, dict) else source
                            if source.startswith('data:image/'):
                                raw = base64.b64decode(source.split(',', 1)[1], validate=True)
                            else:
                                raw, _, _, _ = fetch_public(source, max_bytes=32 * 1024 * 1024)
                            path = Path(directory) / f'{len(list(Path(directory).iterdir()))}.png'
                            path.write_bytes(raw)
                            blocks.append({'type': 'image', 'image': str(path), 'max_pixels': 2000000})
                        else:
                            raise ValueError('Unsupported model input content block')
                    content = blocks
                converted.append({'role': role, 'content': content})
            generation = self.model.generate(converted, tools or [], max_tokens, 65536,
                                             repetition_penalty=1.1)
            status = generation.get('status')
            if status not in {'ok', 'length'}:
                raise RuntimeError('Local model generation failed: ' + str(status))
            text = generation.get('text') or ''
            calls = []
            if status == 'ok':
                try:
                    action, value = parse_output(text)
                except (ValueError, TypeError):
                    action, value = None, None
                if action == 'tool':
                    names = {t['function']['name'] for t in tools or []}
                    name, args = canonical_call(value['name'], value['arguments'], names)
                    calls = [{'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
                              'function': {'name': name, 'arguments': json.dumps(args)}}]
                    text = text.split('<tool_call>', 1)[0].strip()
                else:
                    answer = final_content(text)
                    if answer:
                        text = f'<{self.answer_tag}>{answer}</{self.answer_tag}>' if self.answer_tag else answer
            prompt = generation.get('input_tokens', 0)
            output = generation.get('new_tokens', 0)
            return ChatCompletion.model_validate({
                'id': 'local_' + uuid.uuid4().hex, 'object': 'chat.completion',
                'created': int(time.time()), 'model': self.path,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text,
                                                  **({'tool_calls': calls} if calls else {})},
                             'finish_reason': 'length' if status == 'length' else 'tool_calls' if calls else 'stop'}],
                'usage': {'prompt_tokens': prompt, 'completion_tokens': output, 'total_tokens': prompt + output}})
