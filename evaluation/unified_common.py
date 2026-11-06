"""Shared non-weight settings for four evaluation conditions."""
from pathlib import Path
import json
import struct

from evaluation.prepare import digest
from evaluation.run import LocalModel

ASSETS = ('config.json', 'generation_config.json', 'preprocessor_config.json',
          'processor_config.json', 'video_preprocessor_config.json', 'tokenizer_config.json',
          'tokenizer.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
          'added_tokens.json', 'chat_template.json', 'chat_template.jinja')
CONDITIONS = ('base_direct', 'base_native', 'sft_native', 'base_grpo_native')
LABELS = {'base_direct': 'Base 直接回答', 'base_native': 'Base＋原生6工具',
          'sft_native': 'SFT＋原生6工具', 'base_grpo_native': 'Base+GRPO＋原生6工具'}


def asset_identity(directory):
    directory = Path(directory)
    for name in ['config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json']:
        if not (directory/name).is_file():
            raise ValueError('Missing shared model asset: ' + name)
    return {name: digest(directory/name) for name in ASSETS if (directory/name).is_file()}


def weight_identity(directory):
    """Read safetensors headers only; never load multi-GB weights onto a CPU/GPU."""
    result, shapes = {}, {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 64 * 1024 * 1024:
                raise ValueError('Unexpected model weight header size')
            header = json.loads(stream.read(size))
        result[path.name] = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                             'header_sha256': digest_header(header)}
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if name in shapes:
                raise ValueError('Duplicate weight tensor: ' + name)
            shapes[name] = value['shape']
    if not result:
        raise ValueError('No complete safetensors weights: ' + str(directory))
    return result, shapes


def digest_header(header):
    import hashlib
    return hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()


class SharedLocalModel(LocalModel):
    def __init__(self, weights, assets):
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText, GenerationConfig
        if not torch.cuda.is_available():
            raise RuntimeError('The unified evaluation needs its allocated CUDA GPU')
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.device = torch, 'cuda'
        self.processor = AutoProcessor.from_pretrained(str(assets), local_files_only=True,
                                                       min_pixels=65536, max_pixels=262144)
        config = AutoConfig.from_pretrained(str(assets), local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(weights), config=config, dtype=torch.bfloat16, device_map='cuda',
            attn_implementation='sdpa', local_files_only=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(str(assets), local_files_only=True)


def direct_system(session_module):
    # Preserve the common evidence/answer instructions, removing tool-only directions.
    text = session_module.system_prompt().split('Project compatibility tools', 1)[0].strip()
    text = text.replace('Call tools using <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>. ', '')
    return text + '\nNo tools are available. Use only the question and supplied images. Submit your final answer now in <response>...</response>.'
