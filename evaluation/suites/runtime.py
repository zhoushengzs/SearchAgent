"""Shared project-local locations for upstream benchmark dependencies."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / 'official_eval'


def official_root():
    return Path(os.environ.get('EVAL_OFFICIAL_ROOT', str(DEFAULT_ROOT))).expanduser().resolve()
