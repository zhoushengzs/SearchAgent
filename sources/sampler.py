'Reproducible sampling helpers.'
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from core.utils import get_logger

logger = get_logger(__name__)

Sample = dict[str, Any]


def random_sample(
    samples: list[Sample],
    sample_size: int | None,
    seed: int,
) -> list[Sample]:
    'Global reproducible random sample.'
    if not sample_size or sample_size >= len(samples):
        if sample_size and sample_size > len(samples):
            logger.warning(
                "Requested sample_size=%d but only %d valid samples available; "
                "returning all.",
                sample_size,
                len(samples),
            )
        return list(samples)

    rng = random.Random(seed)
    chosen = rng.sample(samples, sample_size)
    return chosen


def parse_quota(spec: str | None) -> dict[str, int]:
    'Parse a ``--source-quota`` spec into ``{source: cap}``.'
    quota: dict[str, int] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Bad --source-quota entry {part!r}; expected 'name=N' "
                "(e.g. 'mmsearch_plus=150,default=100')"
            )
        name, _, value = part.partition("=")
        name = name.strip()
        try:
            cap = int(value)
        except ValueError as exc:
            raise ValueError(f"Bad cap in --source-quota entry {part!r}") from exc
        if cap < 0:
            raise ValueError(f"Negative cap in --source-quota entry {part!r}")
        quota[name] = cap
    return quota


def per_source_sample(
    samples: list[Sample],
    per_source_size: int | None = None,
    seed: int = 0,
    source_key: str = "source_dataset",
    quota: dict[str, int] | None = None,
) -> list[Sample]:
    'Cap how many samples each source may contribute.'
    if not quota and not per_source_size:
        return list(samples)

    quota = dict(quota or {})
    fallback = quota.pop("default", per_source_size)

    rng = random.Random(seed)
    # Group INDICES rather than the dicts themselves: a pool may contain equal
    # (or even identical) records, and identity-based selection would keep or
    # drop all of them together.
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_source[str(s.get(source_key, "unknown"))].append(i)

    keep: set[int] = set()
    for source, idxs in sorted(by_source.items()):
        cap = quota.get(source, fallback)
        if cap is None or len(idxs) <= cap:
            keep.update(idxs)
        else:
            keep.update(rng.sample(idxs, cap))

    return [s for i, s in enumerate(samples) if i in keep]


def source_distribution(
    samples: list[Sample],
    source_key: str = "source_dataset",
) -> dict[str, int]:
    """Count samples per source_dataset."""
    dist: dict[str, int] = defaultdict(int)
    for s in samples:
        dist[str(s.get(source_key, "unknown"))] += 1
    return dict(sorted(dist.items()))
