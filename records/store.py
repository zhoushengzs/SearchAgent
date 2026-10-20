'Record-level store semantics: success, resume keys, compaction.'
from __future__ import annotations

from typing import Any


def is_successful_record(record: dict[str, Any]) -> bool:
    """A record counts as done only if it produced a valid annotation."""
    return (
        record.get("api_error") is None
        and record.get("parse_success") is True
        and not record.get("validation_errors")
    )


def latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last attempted revision wins, including failures; never revive stale passes."""
    return list({resume_key(r): r for r in records}.values())


def resume_key(record: dict[str, Any]) -> tuple[str, str]:
    'The identity a labeled record resumes under: ``(id, trace_variant)``.'
    return (
        str(record.get("id", "")),
        str(record.get("trace_variant") or "default"),
    )
