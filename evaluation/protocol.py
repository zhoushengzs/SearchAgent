"""Shared training-format protocol for SFT and Base evaluation.

The SFT model was trained on step-wise decisions that include a refreshed state
(attached_images, executed_prefix_step_ids, remaining_decisions), the exact
training system prompt and the training tool schemas.  This module reproduces
that input distribution so both models are evaluated under one protocol.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAINING_SYSTEM = (BASE / 'training_protocol' / 'system.txt').read_text(encoding='utf-8').strip()
TRAINING_TOOLS = json.loads((BASE / 'training_protocol' / 'tools.json').read_text(encoding='utf-8'))
# The state recorded in the SFT data listed these capability names; keep them
# verbatim so the state distribution matches training.  Execution still uses
# the real project tools through the mapping below.
STATE_AVAILABLE_TOOLS = ['abstain', 'calculate', 'crop', 'finish', 'image_search',
                         'inspect_image', 'visit', 'web_search']
STATE_HEADER = ('Recorded task state follows. Historical hypotheses and observer notes can be wrong; '
                'use them as context, not ground truth. Choose the next action from the evidence '
                'available at this decision.\n')


def training_system() -> str:
    return TRAINING_SYSTEM


def training_tools() -> list:
    return copy.deepcopy(TRAINING_TOOLS)


def state_message(task, tools, initial_views, attached, executed, step, max_steps, max_pixels):
    """Build the same user-message shape the SFT data used at every decision."""
    content = []
    for index, image in enumerate(attached, 1):
        content.append({'type': 'text', 'text': f'img_{index}: '})
        content.append({'type': 'image', 'image': image['path'], 'max_pixels': max_pixels})
        content.append({'type': 'text', 'text': '\n'})
    configuration = dict(tools.configuration())
    configuration['available_tools'] = list(STATE_AVAILABLE_TOOLS)
    state = {
        'question': task['model_input']['question'],
        'initial_views': initial_views,
        'image_inventory': tools.inventory(),
        'tool_configuration': configuration,
        'attached_images': [{k: v for k, v in image.items() if k not in ('path', 'artifact')} for image in attached],
        'remaining_decisions': max(1, max_steps - step + 1),
        'executed_prefix_step_ids': list(executed),
    }
    content.append({'type': 'text', 'text': STATE_HEADER + json.dumps(state, ensure_ascii=False)})
    return {'role': 'user', 'content': content}


def observation_message(step: int, result: dict) -> dict:
    """Training observations are {step, result, observation}; no budget wrapper."""
    return {'role': 'tool', 'content': json.dumps(
        {'step': step, 'result': result, 'observation': None}, ensure_ascii=False)}


def _dimensions(attached, reference):
    """Resolve an image reference (img_2 or image_1) to its asset and size."""
    if not isinstance(reference, str) or not attached:
        return None, None
    if reference.startswith('img_'):
        try:
            image = attached[int(reference[4:]) - 1]
        except (ValueError, IndexError):
            return None, None
    else:
        image = next((item for item in attached if item.get('asset_id') == reference), None)
    if not image:
        return None, None
    width, height = image.get('width'), image.get('height')
    if not width or not height:
        return None, None
    return image.get('asset_id'), (int(width), int(height))


def normalize_call(name: str, arguments: dict, attached: list):
    """Map training tool names/arguments onto the executable evaluator tools."""
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError('Tool call must contain a name and an arguments object')
    if name.startswith('vs_'):
        return name, arguments
    if name in {'web_search', 'text_search'}:
        query = arguments.get('q') or arguments.get('query')
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f'{name} needs a non-empty text query in q/query')
        return 'vs_web_search', {'query': query}
    if name == 'visit':
        url = arguments.get('url')
        if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            raise ValueError('visit needs an exact public http(s) URL')
        mapped = {'url': url}
        if arguments.get('offset') is not None:
            mapped['offset'] = arguments['offset']
        return 'vs_visit', mapped
    if name == 'inspect_image':
        if set(arguments) not in ({'asset_id'}, {'image_url'}):
            raise ValueError('inspect_image needs exactly one of asset_id or image_url')
        return 'vs_inspect_image', dict(arguments)
    if name == 'calculate':
        expression = arguments.get('expression')
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError('calculate needs a non-empty expression')
        return 'vs_calculate', {'expression': expression}
    if name == 'crop':
        asset_id, size = _dimensions(attached, arguments.get('image'))
        keys = {'x', 'y', 'width', 'height'}
        if not asset_id or not size or not keys <= set(arguments):
            raise ValueError('crop(image,x,y,width,height) cannot be resolved to a known image; '
                             'use vs_crop with a normalized [x1,y1,x2,y2] bbox on the original asset')
        width, height = size
        x, y = float(arguments['x']), float(arguments['y'])
        w, h = float(arguments['width']), float(arguments['height'])
        bbox = [max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height)),
                max(0.0, min(1.0, (x + w) / width)), max(0.0, min(1.0, (y + h) / height))]
        return 'vs_crop', {'asset_id': asset_id, 'bbox': bbox}
    if name == 'image_search':
        raise ValueError('native image_search (reverse visual search) is unavailable in this evaluator; '
                         'use vs_image_search with a text query, or vs_inspect_image with a public image URL')
    if name == 'layout_parsing':
        raise ValueError('layout_parsing (OCR) is unavailable; use vs_inspect_image or vs_visit instead')
    raise ValueError(f'Unknown tool name {name!r}; use one of the supplied tools')
