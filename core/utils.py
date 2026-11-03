"""Small, dependency-light helpers shared across the pipeline."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured to print to stderr once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# IDs / hashing
# --------------------------------------------------------------------------- #
def stable_hash(*parts: Any, length: int = 10) -> str:
    """Deterministic short hash from arbitrary string-able parts."""
    joined = "||".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return digest[:length]


def make_stable_id(source_dataset: str, index: int, *parts: Any) -> str:
    """Build a reproducible sample id: <source>_<index>_<hash>."""
    return f"{source_dataset}_{index:06d}_{stable_hash(*parts)}"


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def read_image_size(image_path: str | Path) -> dict[str, int] | None:
    'Real pixel size of an image (header-only read), or None if unreadable.'
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
        return {"width": int(w), "height": int(h)}
    except Exception:  # noqa: BLE001 - missing file / not an image
        return None


def image_to_data_uri(image_path: str | Path) -> str:
    """Send the executor's recorded view byte-for-byte; never alter evidence."""
    from PIL import Image

    path = Path(image_path)
    with Image.open(path) as image:
        suffix = (image.format or "").lower()
        if max(image.size) > 2048 or suffix not in {"png", "jpeg", "webp", "gif"}:
            raise ValueError("Supply an executed view of at most 2048 pixels")
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Executed view exceeds the image transfer budget")
    return f"data:image/{suffix};base64,{base64.b64encode(raw).decode('ascii')}"


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory of ``path`` if needed; return the Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #
def json_safe(value: Any, _depth: int = 0, _max_depth: int = 12) -> Any:
    'Recursively convert a value into something ``json.dumps`` can handle.'
    from core.config import IMAGE_BYTES_PLACEHOLDER

    if _depth > _max_depth:
        return str(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return IMAGE_BYTES_PLACEHOLDER
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            # Common image byte carriers -> strip.
            if isinstance(v, bytes):
                out[key] = IMAGE_BYTES_PLACEHOLDER
            else:
                out[key] = json_safe(v, _depth + 1, _max_depth)
        return out
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v, _depth + 1, _max_depth) for v in value]

    # PIL image or other objects -> describe rather than serialize.
    type_name = type(value).__name__
    if type_name == "Image" or "PIL" in type(value).__module__:
        return IMAGE_BYTES_PLACEHOLDER
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)
