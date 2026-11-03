"""Isolated evidence and image numbering for one model answer."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from PIL import Image

from .evidence import EvidenceTools, TOOL_NAMES
from .identity import file_sha

HERE = Path(__file__).parent
SCHEMAS = json.loads((HERE / 'tools.json').read_text())
STATE_HEADER = ('Recorded task state follows. Historical hypotheses and observer notes can be wrong; '
                'use them as context, not ground truth. Choose the next action from the evidence '
                'available at this decision.\n')


def system_prompt():
    return ((HERE / 'system.txt').read_text().strip() + '\n\n# Tools\n<tools>\n' +
            '\n'.join(json.dumps(s, ensure_ascii=False) for s in SCHEMAS) + '\n</tools>')


def get_all_tools():
    return {s['function']['name']: copy.deepcopy(s) for s in SCHEMAS}


def public_result(value):
    """Do not expose host filenames; retain the actual returned evidence."""
    if isinstance(value, dict):
        return {k: public_result(v) for k, v in value.items()
                if k not in {'path', 'source_path', 'raw_artifact', 'artifact'}}
    if isinstance(value, list):
        return [public_result(v) for v in value]
    return value


def parse_action(content):
    # Accept one complete action anywhere in the reply. The <think> block is
    # optional and free-form reasoning text around the action is tolerated, so
    # that base-model answers still execute and receive answer-based reward;
    # the format reward separately scores whether <think> was present.
    if not isinstance(content, str) or not content:
        raise ValueError('Empty model response.')
    body = re.sub(r'<think>.*?</think>', '', content, flags=re.S)
    tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', body, re.S)
    responses = re.findall(r'<response>(.*?)</response>', body, re.S)
    if len(tool_calls) + len(responses) != 1:
        raise ValueError('Use one <tool_call> or <response> (optionally preceded by <think>).')
    if body.count('<tool_call>') != len(tool_calls) or body.count('</tool_call>') != len(tool_calls):
        raise ValueError('Incomplete tool call tag.')
    if body.count('<response>') != len(responses) or body.count('</response>') != len(responses):
        raise ValueError('Incomplete response tag.')
    if responses:
        if not responses[0].strip():
            raise ValueError('The final answer is empty.')
        return 'answer', responses[0].strip()
    payload = tool_calls[0].strip()
    call = json.loads(payload)
    if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
        raise ValueError('A tool call needs name and arguments.')
    if call['name'] not in TOOL_NAMES or not isinstance(call['arguments'], dict):
        raise ValueError('Use one of the six supplied tools and an arguments object.')
    return 'tool', call


class ProjectSession:
    def __init__(self, question, images, directory, max_steps):
        self.question = question
        self.directory = Path(directory)
        self.tools = EvidenceTools(self.directory, page_chars=4000)
        self.max_steps = max_steps
        self.executed = []
        self.successful_calls = {}
        self.attached = []
        inputs = []
        for index, image in enumerate(images or [], 1):
            # Persist originals before making 512-pixel previews. Crops always
            # use these originals, not the already resized preview.
            target = self.directory / f'original_{index}.png'
            if isinstance(image, Image.Image):
                image.convert('RGB').save(target)
            else:
                with Image.open(image) as source:
                    source.convert('RGB').save(target)
            inputs.append({'index': index, 'path': str(target), 'sha256': file_sha(target)})
        self.initial_views = self.tools.initialize(inputs)
        for view in self.initial_views:
            self._attach(view, 0)
        # Follow the existing rule: visiting an URL explicitly supplied in the
        # question is permitted, as well as URLs returned by real searches.
        for url in re.findall(r'https?://[^\s<>"\x27]+', question):
            self.tools.urls.add(url.rstrip('.,);]'))

    def _attach(self, view, step):
        item = dict(view, image_index=len(self.attached) + 1, observed_at_step=step)
        self.attached.append(item)
        with Image.open(item['path']) as image:
            return image.convert('RGB').copy()

    def state(self, step):
        return {'question': self.question, 'initial_views': public_result(self.initial_views),
                'image_inventory': self.tools.inventory(),
                'tool_configuration': self.tools.configuration(),
                'attached_images': public_result(self.attached),
                'remaining_decisions': max(0, self.max_steps - step + 1),
                'executed_prefix_step_ids': list(self.executed)}

    def message(self, step, *, result=None, new_images=None, labels=None):
        text = ''
        if result is not None:
            text += '<tool_response>\n' + json.dumps(public_result(result), ensure_ascii=False) + '\n</tool_response>\n'
        if labels:
            text += 'Images attached to this message, in order: ' + ', '.join(labels) + '.\n'
        text += STATE_HEADER + json.dumps(self.state(step), ensure_ascii=False)
        if step >= self.max_steps:
            text += '\nThis is your final decision. Do not call another tool. Give the best supported final answer in <response>...</response>; state uncertainty if evidence is insufficient.'
        message = {'role': 'user', 'content': text}
        if new_images:
            message['images'] = new_images
        return message

    def initial_message(self):
        images = []
        for view in self.attached:
            with Image.open(view['path']) as image:
                images.append(image.convert('RGB').copy())
        return self.message(1, new_images=images,
                            labels=[f'img_{v["image_index"]}' for v in self.attached])

    def execute(self, name, arguments, step):
        if name not in TOOL_NAMES:
            raise ValueError('Unknown project tool')
        if not isinstance(arguments, dict):
            raise ValueError('arguments must be an object')
        schema = next(s['function']['parameters'] for s in SCHEMAS if s['function']['name'] == name)
        if set(arguments) - set(schema['properties']) or not set(schema.get('required', [])) <= set(arguments):
            raise ValueError('Tool arguments do not match the supplied tool description')
        if name == 'vs_inspect_image' and set(arguments) not in ({'asset_id'}, {'image_url'}):
            raise ValueError('Use exactly one of asset_id and image_url')
        call_key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False))
        if call_key in self.successful_calls:
            previous_step = self.successful_calls[call_key]
            result = {'tool': name, 'status': 'error', 'error_type': 'RepeatedAction',
                      'message': f'This exact successful call already ran at step {previous_step}. No new request was sent. Read that result; use a different query, visit an acquired URL, inspect an image, or provide <think>brief reasoning</think><response>your answer</response>.'}
            self.executed.append(step)
            return result, self.message(step + 1, result=result)
        result = self.tools.execute(name.removeprefix('vs_'), arguments, step)
        if result.get('status') == 'ok':
            self.successful_calls[call_key] = step
        result['tool'] = name
        self.executed.append(step)
        images, labels = [], []
        view = result.get('data', {}).get('view') if result.get('status') == 'ok' else None
        if view:
            view = dict(view, asset_id=result['data']['asset_id'])
            images.append(self._attach(view, step))
            labels.append(f'img_{self.attached[-1]["image_index"]}')
        return result, self.message(step + 1, result=result, new_images=images, labels=labels)
