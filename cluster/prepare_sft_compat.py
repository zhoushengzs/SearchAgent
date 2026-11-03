"""Expose HF5-saved SFT assets to the verified HF4 training environment.
Weights and tokenizer.json remain the original files; only JSON layouts change.
"""
import argparse, hashlib, json
from pathlib import Path
BASE = Path('/cluster/home/shenqunlun/zhousheng')

def prepare():
    source = BASE / 'qwen3-8b-sft'
    target = BASE / 'qwen3-8b-sft-transformers4'
    target.mkdir(exist_ok=True)
    def write(name, data):
        path = target / name
        content = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    for f in source.iterdir():
        if f.is_file() and f.name not in {'config.json', 'tokenizer_config.json', 'processor_config.json'}:
            dest = target / f.name
            if dest.is_symlink():
                assert dest.resolve() == f.resolve()
            elif dest.exists():
                raise RuntimeError('Unexpected existing compatibility file: ' + str(dest))
            else:
                dest.symlink_to(f)
    config = json.loads((source / 'config.json').read_text())
    text = config['text_config']
    if 'rope_parameters' in text:
        rope = dict(text.pop('rope_parameters'))
        text['rope_theta'] = rope.pop('rope_theta')
        text['rope_scaling'] = rope
    write('config.json', config)
    tok = json.loads((source / 'tokenizer_config.json').read_text())
    extra = tok.get('extra_special_tokens')
    if isinstance(extra, list):
        tok.pop('extra_special_tokens')
        tok['additional_special_tokens'] = list(dict.fromkeys(tok.get('additional_special_tokens', []) + extra))
    raw = json.loads((source / 'tokenizer.json').read_text())
    tok['added_tokens_decoder'] = {str(t['id']): {k:v for k,v in t.items() if k != 'id'} for t in raw['added_tokens']}
    write('tokenizer_config.json', tok)
    proc = json.loads((source / 'processor_config.json').read_text())
    for key, name in [('image_processor','preprocessor_config.json'), ('video_processor','video_preprocessor_config.json')]:
        value = dict(proc.pop(key))
        value['processor_class'] = proc['processor_class']
        write(name, value)
    write('processor_config.json', proc)
    write('compatibility.json', {'source': str(source), 'original_assets_unchanged': True,
          'source_configs_sha256': {n:hashlib.sha256((source/n).read_bytes()).hexdigest()
           for n in ['config.json','tokenizer_config.json','processor_config.json','tokenizer.json']}})
    return target

def validate(target):
    from transformers import AutoConfig, AutoTokenizer, AutoProcessor
    from PIL import Image
    source = BASE / 'qwen3-8b-sft'
    config = AutoConfig.from_pretrained(target, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    processor = AutoProcessor.from_pretrained(target, local_files_only=True)
    raw = json.loads((source / 'tokenizer.json').read_text())
    expected_vocab = dict(raw['model']['vocab'])
    expected_vocab.update({t['content']:t['id'] for t in raw['added_tokens']})
    assert tokenizer.get_vocab() == expected_vocab, 'Vocabulary or token IDs changed'
    assert tokenizer.eos_token_id == 151645 and tokenizer.pad_token_id == 151643
    assert config.text_config.rope_theta == 5000000
    assert config.text_config.rope_scaling['mrope_section'] == [24,20,20]
    original = json.loads((source/'processor_config.json').read_text())['image_processor']
    for key in ['patch_size','temporal_patch_size','merge_size','image_mean','image_std']:
        assert getattr(processor.image_processor,key) == original[key], key
    assert processor.image_processor.size == original['size']
    encoded = processor(images=[Image.new('RGB',(64,64),'white')],
                        text=['<|vision_start|><|image_pad|><|vision_end|>Describe this image.'], return_tensors='pt')
    assert encoded['pixel_values'].numel() > 0
    assert (target/'model.safetensors').resolve() == (source/'model.safetensors').resolve()
    print('SFT_COMPAT_OK', 'vocab',len(expected_vocab),'image_grid',encoded['image_grid_thw'].tolist(), flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');a=p.parse_args()
    target=prepare()
    if a.validate: validate(target)
    print(target,flush=True)
