"""RECOIL advantage composition."""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class RecoilTrainerConfig:
    """Read from RECOIL_<FIELD>."""

    enable: bool = False
    cv: bool = True
    cv_shrink: float = 0.5
    cf_weight: float = 1.0

    @classmethod
    def from_env(cls, environ=None) -> "RecoilTrainerConfig":
        env = os.environ if environ is None else environ
        values = {}
        for f in fields(cls):
            raw = env.get("RECOIL_" + f.name.upper())
            if raw not in (None, ""):
                default = getattr(cls, f.name)
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on") if isinstance(default, bool) else float(raw)
        config = cls(**values)
        if not 0 <= config.cv_shrink <= 1 or config.cf_weight < 0:
            raise ValueError("RECOIL_CV_SHRINK must lie in [0, 1] and RECOIL_CF_WEIGHT must be >= 0")
        return config


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def turn_values(metas, group_ids, cfg: RecoilTrainerConfig):
    """Cross-fitted per-turn advantages."""
    groups = defaultdict(list)
    for i, meta in enumerate(metas):
        if meta is not None:
            groups[group_ids[i]].append(i)
    stats, points = {}, defaultdict(list)
    for group, rows in groups.items():
        roots = {metas[i]["episode"]: metas[i]["ret"][0] for i in rows if metas[i]["kind"] == "main" and metas[i]["ret"]}
        leaves, beliefs = defaultdict(list), defaultdict(list)
        for i in rows:
            meta = metas[i]
            leaves[meta["episode"]].append(meta["leaf"])
            if meta.get("aux") is None:
                beliefs[meta["episode"]] += [x for t, (w, x) in enumerate(zip(meta["w"], meta["x"]))
                                             if w > 0 and x is not None and t >= meta["trainable_from"]]
        for episode in leaves:
            stats[group, episode] = {
                "base": _mean([r for e, r in roots.items() if e != episode]),
                "scale": _std([v for e, vs in leaves.items() if e != episode for v in vs]) or 1.0,
                "mx": _mean([x for e, xs in beliefs.items() if e != episode for x in xs]),
            }
        for i in rows:
            meta = metas[i]
            if meta.get("aux") is not None:
                continue
            stat = stats[group, meta["episode"]]
            for t, (w, r, x) in enumerate(zip(meta["w"], meta["ret"], meta["x"])):
                if w > 0 and x is not None and t >= meta["trainable_from"]:
                    points[group].append((x - stat["mx"], r - stat["base"]))
    slopes = {}
    if cfg.cv:
        for group in groups:
            other = [p for g, ps in points.items() if g != group for p in ps]
            var = sum(x * x for x, _ in other)
            slopes[group] = cfg.cv_shrink * sum(x * y for x, y in other) / var if len(other) >= 3 and var > 1e-12 else 0.0
    values = {}
    for group, rows in groups.items():
        slope = slopes.get(group, 0.0)
        for i in rows:
            meta = metas[i]
            stat = stats[group, meta["episode"]]
            if meta.get("aux") is not None:
                values[i] = [cfg.cf_weight * a / stat["scale"] for a in meta["aux"]]
                continue
            values[i] = [w * (r - stat["base"] - (slope * (x - stat["mx"]) if x is not None else 0.0)) / stat["scale"]
                         for w, r, x in zip(meta["w"], meta["ret"], meta["x"])]
    return values, list(slopes.values())


def compose_recoil_advantages(advantages, response_mask, turn_ids, metas, group_ids, cfg: RecoilTrainerConfig, is_fatal=None):
    """Compose per-token advantages."""
    out = advantages.clone()
    if is_fatal is not None:
        legacy = torch.tensor([m is None and bool(f) for m, f in zip(metas, is_fatal)], dtype=torch.bool, device=out.device)
        out[legacy] = out[legacy].clamp(min=0.0)
    rows = [i for i, m in enumerate(metas) if m is not None]
    if not rows:
        return out, {}
    values, slopes = turn_values(metas, group_ids, cfg)
    width = max(len(values[i]) for i in rows)
    table = torch.zeros((len(rows), width + 1), dtype=out.dtype)  # column 0 holds non-assistant tokens
    for r, i in enumerate(rows):
        table[r, 1 : len(values[i]) + 1] = torch.tensor(values[i], dtype=out.dtype)
    index = (turn_ids[rows].to(torch.long) + 1).clamp(0, width).cpu()
    mask = response_mask[rows].to(out.dtype)
    composed = table.gather(1, index).to(out.device) * mask
    clamp = torch.tensor([bool(metas[i].get("clamp_ep")) for i in rows], device=out.device)[:, None]
    composed = torch.where(clamp, composed.clamp(min=0.0), composed)
    out[rows] = composed

    trained = mask.sum(dim=1) > 0
    kinds = [metas[i].get("kind", "main") for i in rows]
    metrics = {f"recoil/rows_{kind}": float(sum(1 for k, t in zip(kinds, trained) if k == kind and t)) for kind in set(kinds)}
    metrics["recoil/rows_clamped"] = float(clamp.sum())
    if slopes:
        metrics["recoil/cv_slope"] = _mean(slopes)
    tokens = float(mask.sum())
    if tokens > 0:
        metrics["recoil/adv_abs"] = float(composed.abs().sum()) / tokens
    return out, metrics
