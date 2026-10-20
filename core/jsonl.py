'Generic JSONL read/write helpers.'
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

from core.utils import ensure_parent_dir, get_logger

logger = get_logger(__name__)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    'Append a single record as one JSON line (ensure_ascii=False).'
    p = ensure_parent_dir(path)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")
        fh.flush()


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file; retain the old file if writing fails."""
    p = ensure_parent_dir(path)
    fd, temporary = tempfile.mkstemp(prefix=p.name + ".", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, p)
    finally:
        Path(temporary).unlink(missing_ok=True)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, skipping blank/broken lines."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    yield record
                else:
                    logger.warning("Skipping non-object line %d in %s", lineno, p)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, p, exc)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read an entire JSONL file into a list of dicts."""
    return list(iter_jsonl(path))
