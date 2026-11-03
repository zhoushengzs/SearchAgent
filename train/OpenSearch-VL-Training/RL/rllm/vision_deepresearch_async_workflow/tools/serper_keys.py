"""Ordered Serper credentials with shared, secret-free exhaustion state.

SERPER_API_KEYS accepts a JSON list. Legacy SERPER_API_KEY[_N] and
SERP_API_KEY[_N] are accepted when the JSON list is absent. Linux workers
on the same host coordinate with flock; only explicit credit exhaustion
advances the index. No API key is written to the state file or log.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger(__name__)


class SerperKeysExhausted(SystemExit):
    """Abort the current worker rather than fabricate a search result."""


def configured_keys(environ=None):
    env = os.environ if environ is None else environ
    raw = env.get("SERPER_API_KEYS", "").strip()
    if raw:
        values = json.loads(raw)
        if not isinstance(values, list) or not values or any(
            not isinstance(v, str) or not v.strip() for v in values
        ):
            raise ValueError("SERPER_API_KEYS must be a nonempty JSON list of strings")
    else:
        values = [env.get("SERPER_API_KEY") or env.get("SERP_API_KEY")]
        indices = sorted({int(m.group(1)) for name in env
                          if (m := re.fullmatch(r"(?:SERPER|SERP)_API_KEY_(\d+)", name))})
        values.extend(env.get(f"SERPER_API_KEY_{i}") or env.get(f"SERP_API_KEY_{i}")
                      for i in indices)
    return tuple(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def credits_exhausted(response):
    if response.status_code < 400:
        return False
    text = response.text[:2048].lower()
    if response.status_code == 429 and not any(w in text for w in ("credit", "balance")) and any(w in text for w in ("rate limit", "per second", "per minute")):
        return False
    return (any(w in text for w in ("credit", "quota", "balance")) and
            any(w in text for w in ("exhaust", "insufficient", "not enough",
                                    "out of", "deplet", "exceed", "no more", "zero")))


class SerperKeyPool:
    def __init__(self, keys, state_file):
        self.keys = tuple(dict.fromkeys(keys))
        if not self.keys:
            raise RuntimeError("Configure SERPER_API_KEYS or SERPER_API_KEY")
        self.path = Path(state_file)
        self.fingerprint = hashlib.sha256(json.dumps(self.keys).encode()).hexdigest()
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def _state(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    raw = f.read()
                    state = json.loads(raw) if raw else {}
                    if state.get("pool") != self.fingerprint:
                        state = {"pool": self.fingerprint, "index": 0,
                                 "key_count": len(self.keys)}
                    if type(state.get("index")) is not int or not 0 <= state["index"] <= len(self.keys):
                        raise ValueError("Invalid Serper state index")
                    yield state
                    f.seek(0)
                    json.dump(state, f)
                    f.truncate()
                    f.flush()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def current(self):
        with self._state() as state:
            index = state["index"]
        if index >= len(self.keys):
            raise SerperKeysExhausted("All configured Serper keys exhausted; stop and refill credits")
        return index, self.keys[index]

    def exhausted(self, index):
        with self._state() as state:
            previous = state["index"]
            state["index"] = max(previous, index + 1)
            current = state["index"]
        if current > previous:
            log.warning("Serper credits exhausted for key %d/%d; next position %d",
                        index + 1, len(self.keys), current + 1)


_pool = None
_pool_config = None
_pool_lock = threading.Lock()


def key_pool():
    global _pool, _pool_config
    keys = configured_keys()
    root = os.environ.get("GRPO_DATA_ROOT", str(Path(__file__).resolve().parents[4]))
    state = os.environ.get("SERPER_KEY_STATE_FILE", str(Path(root) / ".codex/serper_key_state.json"))
    config = (keys, state)
    with _pool_lock:
        if config != _pool_config:
            _pool = SerperKeyPool(keys, state)
            _pool_config = config
        return _pool


def serper_post(url, *, headers=None, **kwargs):
    import requests

    headers = dict(headers or {})
    # Preserve the existing optional gateway route without sending direct keys there.
    if any(k.lower() == "authorization" for k in headers):
        return requests.post(url, headers=headers, **kwargs)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "google.serper.dev" or parsed.port not in (None, 443) or parsed.username or parsed.password:
        raise ValueError("Direct Serper credentials require https://google.serper.dev")
    pool = key_pool()
    while True:
        index, key = pool.current()
        request_headers = {k: v for k, v in headers.items() if k.lower() != "x-api-key"}
        request_headers["X-API-KEY"] = key
        for attempt in range(3):
            response = requests.post(url, headers=request_headers, **kwargs)
            if credits_exhausted(response):
                pool.exhausted(index)
                break
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
            return response

