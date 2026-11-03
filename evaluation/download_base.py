"""Download into ~/models. Completed files are reused; this Hub version restarts partial files."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id', default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--output', type=Path, default=Path.home() / 'models/Qwen3-VL-8B-Instruct')
    a = p.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(a.model_id, local_dir=str(a.output),
                      allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.txt', '*.model'],
                      max_workers=2)
    print(f'Original model saved to: {a.output.resolve()}')


if __name__ == '__main__':
    main()
