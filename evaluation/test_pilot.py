"""Offline checks of fixed routing and the local benchmark transport."""
import base64
import io
from pathlib import Path
import unittest

from PIL import Image

from evaluation.local_api import LocalCompletionClient, canonical_call
from evaluation.pilot import selection


class PilotTests(unittest.TestCase):
    def test_sampling_is_stable_per_benchmark_and_rounds_up(self):
        rows = [{'id': f'{source}_{i}', 'source_dataset': source, 'input_fingerprint': str(i)}
                for source, n in [('agentvista', 33), ('mmbrowsecomp', 88), ('other', 59)] for i in range(n)]
        selected = selection(rows)
        reversed_selected = selection(list(reversed(rows)))
        self.assertEqual({k: set(v) for k, v in selected.items()}, {k: set(v) for k, v in reversed_selected.items()})
        self.assertEqual({k: len(v) for k, v in selected.items()}, {'agentvista': 7, 'mmbrowsecomp': 18, 'other': 12})

    def test_only_lossless_tool_aliases_are_mapped(self):
        self.assertEqual(canonical_call('vs_image_search', {'query': 'red car'}, {'image_search'}),
                         ('image_search', {'query': 'red car', 'search_type': 'text'}))
        self.assertEqual(canonical_call('vs_crop', {}, {'code_interpreter'}), ('vs_crop', {}))

    def test_local_transport_preserves_pixels_tools_and_final_answer(self):
        class Model:
            seen = []
            def generate(self, messages, tools, max_tokens, context_tokens, **kwargs):
                self.seen.append((messages, tools))
                if len(self.seen) == 1:
                    path = messages[0]['content'][0]['image']
                    with Image.open(path) as img:
                        assert img.getpixel((0, 0)) == (255, 0, 0)
                    return {'text': '<tool_call>{"name":"vs_web_search","arguments":{"query":"q"}}</tool_call>',
                            'status': 'ok', 'input_tokens': 4, 'new_tokens': 8}
                return {'text': '<response>42</response>', 'status': 'ok', 'input_tokens': 5, 'new_tokens': 2}
        stream = io.BytesIO(); Image.new('RGB', (32, 32), 'red').save(stream, format='PNG')
        source = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
        model = Model(); client = LocalCompletionClient('/tmp/fixture', model=model, answer_tag='answer')
        result = client.create(messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': source}}]}],
                               tools=[{'type': 'function', 'function': {'name': 'web_search'}}])
        self.assertEqual(result.choices[0].message.tool_calls[0].function.name, 'web_search')
        self.assertEqual(result.choices[0].finish_reason, 'tool_calls')
        result = client.create(messages=[{'role': 'user', 'content': 'answer'}])
        self.assertEqual(result.choices[0].message.content, '<answer>42</answer>')


if __name__ == '__main__':
    unittest.main()
