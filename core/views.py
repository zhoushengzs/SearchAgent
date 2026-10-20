"""Materialize the exact images seen by the sequential observation policy."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.provenance import file_sha


def valid_bbox(bbox: Any) -> bool:
    return (isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and 0 <= v <= 1 for v in bbox)
            and bbox[0] < bbox[2] and bbox[1] < bbox[3])


def view_identity(view: dict[str, Any]) -> tuple:
    """Same original region/resolution/bytes; null bbox denotes the whole image."""
    return (tuple(view["bbox"] or [0, 0, 1, 1]), view["width"], view["height"], view["sha256"])


def _render_view(image_path: str, bbox: list[float] | None, max_side: int):
    from PIL import Image, ImageOps

    # Bound RGB PNG payloads below the client's per-image transfer threshold,
    # so the recorded view is not silently resized or re-encoded on submission.
    if not 1 <= max_side <= 2048:
        raise ValueError("Executed view longest side must be between 1 and 2048")

    with Image.open(image_path) as image:
        # Coordinate space matches read_image_size: do not rotate via EXIF.
        image = image.convert("RGB")
        w, h = image.size
        if bbox is not None:
            if not valid_bbox(bbox):
                raise ValueError(f"Invalid normalized bbox: {bbox}")
            pixels = (math.floor(bbox[0] * w), math.floor(bbox[1] * h),
                      math.ceil(bbox[2] * w), math.ceil(bbox[3] * h))
            if pixels[2] - pixels[0] < 8 or pixels[3] - pixels[1] < 8:
                raise ValueError("Crop is smaller than 8 pixels; cannot recover detail")
            image = image.crop(pixels)
        else:
            pixels = (0, 0, w, h)
        side = min(max_side, max(image.size))
        image = ImageOps.contain(image, (side, side),
                                 Image.Resampling.LANCZOS)
        return image, list(pixels)


def materialize_view(image_path: str, output: Path, bbox: list[float] | None,
                     max_side: int) -> dict[str, Any]:
    image, pixels = _render_view(image_path, bbox, max_side)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    image.save(temp, format="PNG")
    temp.replace(output)
    return {"path": str(output.resolve()), "sha256": file_sha(output),
            "bbox": bbox, "pixel_bbox": pixels, "width": image.width, "height": image.height}


def matches_materialized_view(image_path: str, view: dict, bbox: list[float] | None,
                              max_side: int) -> bool:
    """Check recorded coordinates and actual pixels against the executed crop."""
    from PIL import Image

    expected, pixels = _render_view(image_path, bbox, max_side)
    with Image.open(view["path"]) as actual:
        return (view["bbox"] == bbox and view["pixel_bbox"] == pixels and
                (view["width"], view["height"]) == expected.size == actual.size and
                expected.tobytes() == actual.convert("RGB").tobytes())
