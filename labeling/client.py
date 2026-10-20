'Vision-language client (DashScope or OpenRouter) with retries.'
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_MAX_TOKENS,
    PROVIDERS,
    DEFAULT_RETRY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    TRUNCATION_RETRY_MAX_TOKENS,
)
from core.utils import get_logger, image_to_data_uri

logger = get_logger(__name__)


@dataclass
class GenResult:
    'Outcome of a single generation.'

    text: str
    finish_reason: str | None = None
    truncated: bool = False


def _read_api_key(provider: str) -> str | None:
    for var in PROVIDERS[provider]["key_env"]:
        val = os.environ.get(var)
        if val:
            return val
    return None


def resolve_provider(provider: str = "auto") -> str:
    """``auto`` -> the first provider in ``PROVIDERS`` whose key is set."""
    if provider != "auto":
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; "
                             f"choose from {sorted(PROVIDERS)}")
        return provider
    for name in PROVIDERS:
        if _read_api_key(name):
            return name
    wanted = " / ".join(v for p in PROVIDERS.values() for v in p["key_env"])
    raise RuntimeError(f"No API key found. Set one of: {wanted}")


def _is_throttle_error(exc: Exception) -> bool:
    """True if the exception looks like server-side rate limiting (HTTP 429)."""
    s = f"{type(exc).__name__}: {exc}".lower()
    return (
        "429" in s
        or "throttl" in s
        or "rate limit" in s
        or "limit_requests" in s
        or "concurrenc" in s
    )


class VLMClient:
    'Thin wrapper around an OpenAI-compatible vision-language chat API.'

    def __init__(
        self,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        retry: int = DEFAULT_RETRY,
        provider: str = "auto",
    ) -> None:
        self.provider = resolve_provider(provider)
        spec = PROVIDERS[self.provider]
        self.model = model or spec["default_model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry = max(1, retry)
        self.base_url = spec["base_url"]

        self.api_key = _read_api_key(self.provider)
        if not self.api_key:
            raise RuntimeError(
                f"No API key for provider {self.provider!r}. Set one of: "
                + " / ".join(spec["key_env"])
            )
        logger.info("Provider: %s  model: %s", self.provider, self.model)
        self._client = None  # Lazily created real SDK client.
        self._client_lock = threading.Lock()  # Guards lazy _client creation.
        self._request_local = threading.local()

    def generate(self, prompt: str, image_path: str | Path | list[str]) -> GenResult:
        "Return the model's output for (prompt, image) as a ``GenResult``."
        text, finish = self._call_with_retries(prompt, image_path, self.max_tokens)
        if finish != "length":
            return GenResult(text, finish, False)

        bumped = min(self.max_tokens * 2, TRUNCATION_RETRY_MAX_TOKENS)
        if bumped <= self.max_tokens:
            logger.warning("Output truncated (finish_reason=length) and max_tokens "
                           "already at ceiling %d; flagging.", self.max_tokens)
            return GenResult(text, finish, True)

        logger.warning("Output truncated (finish_reason=length); retrying once "
                       "with max_tokens=%d.", bumped)
        text2, finish2 = self._call_with_retries(prompt, image_path, bumped)
        return GenResult(text2, finish2, finish2 == "length")

    def generate_images(self, prompt: str, image_paths: list[str]) -> GenResult:
        """Replay only explicitly supplied views; no source image is added."""
        if not image_paths:
            raise ValueError("At least one image view is required")
        return self.generate(prompt, image_paths)

    def generate_json(self, prompt: str, image_paths: list[str], schema: dict) -> GenResult:
        """Constrain the API response schema without altering the recorded prompt."""
        previous = getattr(self._request_local, "response_format", None)
        self._request_local.response_format = {"type": "json_schema", "json_schema": {
            "name": "visual_agent_response", "strict": True, "schema": schema}}
        try:
            if image_paths:
                return self.generate_images(prompt, image_paths)
            return GenResult(self.generate_text(prompt), "stop", False)
        finally:
            self._request_local.response_format = previous

    def _call_with_retries(
        self, prompt: str, image_path: str | Path | list[str], max_tokens: int
    ) -> tuple[str, str | None]:
        'Call the real API with exponential-backoff retry on network errors.'
        last_exc: Exception | None = None
        for attempt in range(1, self.retry + 1):
            try:
                return self._call_real_api(prompt, image_path, max_tokens)
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_exc = exc
                # Throttling (429) needs a longer cooldown than a transient
                # network blip, else concurrent workers just re-collide.
                throttled = _is_throttle_error(exc)
                backoff = min(30 * 2 ** (attempt - 1), 120) if throttled else (2 ** (attempt - 1))
                logger.warning(
                    "VLM API call failed (attempt %d/%d%s): %s. Backoff %ds.",
                    attempt, self.retry, " THROTTLED" if throttled else "",
                    exc, backoff,
                )
                if attempt < self.retry:
                    time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        'Text-only completion — no image attached.'
        last_exc: Exception | None = None
        for attempt in range(1, self.retry + 1):
            try:
                text, finish = self._call_text_api(prompt, max_tokens)
                if finish == "length":
                    budget = max_tokens if max_tokens is not None else self.max_tokens
                    bumped = min(budget * 2, TRUNCATION_RETRY_MAX_TOKENS)
                    if bumped > budget:
                        text, finish = self._call_text_api(prompt, bumped)
                    if finish == "length":
                        raise ValueError("Text response truncated at token ceiling")
                return text
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_exc = exc
                throttled = _is_throttle_error(exc)
                backoff = min(30 * 2 ** (attempt - 1), 120) if throttled else (2 ** (attempt - 1))
                logger.warning("Text call failed (attempt %d/%d%s): %s. Backoff %ds.",
                               attempt, self.retry, " THROTTLED" if throttled else "",
                               exc, backoff)
                if attempt < self.retry:
                    time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def _sdk(self):
        """Lazily create the OpenAI-compatible SDK client (thread-safe)."""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed. Run `pip install openai`, or "
                "replace _call_real_api with your preferred SDK."
            ) from exc
        if self._client is None:
            with self._client_lock:
                if self._client is None:  # double-checked under lock
                    self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _chat(self, content: list[dict[str, Any]],
              max_tokens: int | None) -> tuple[str, str | None]:
        """One chat completion. Returns ``(text, finish_reason)``."""
        extra = {}
        if self.provider == "openrouter" and self.model in {
            "qwen/qwen3.8-flash", "deepseek/deepseek-v4.1-flash"
        }:
            extra["extra_body"] = {"reasoning": {"effort": "low"}}
        response_format = getattr(self._request_local, "response_format", None)
        if response_format is not None:
            extra["response_format"] = response_format
        response = self._sdk().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            timeout=self.timeout,
            **extra,
        )
        choice = response.choices[0]
        return (choice.message.content or ""), getattr(choice, "finish_reason", None)

    def _call_text_api(self, prompt: str,
                       max_tokens: int | None = None) -> tuple[str, str | None]:
        return self._chat([{"type": "text", "text": prompt}], max_tokens)

    def _call_real_api(
        self, prompt: str, image_path: str | Path | list[str], max_tokens: int | None = None
    ) -> tuple[str, str | None]:
        'Call the real VLM API. Returns ``(content, finish_reason)``.'
        paths = image_path if isinstance(image_path, list) else [image_path]
        images = [{"type": "image_url", "image_url": {"url": image_to_data_uri(p)}}
                  for p in paths]
        return self._chat([
            *images,
            {"type": "text", "text": prompt},
        ], max_tokens)
