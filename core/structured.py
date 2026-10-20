"""Strict JSON responses for decisions and verification (never repair verdicts)."""
from __future__ import annotations

import json
from typing import Any


def json_object(text: str) -> dict[str, Any]:
    body = text.strip()
    if body.startswith("```") and body.endswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    obj = json.loads(body, parse_constant=invalid_constant,
                     object_pairs_hook=unique_pairs)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj
