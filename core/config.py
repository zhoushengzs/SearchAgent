'Global configuration and shared constants for the VLM reasoning labeler.'
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_IMAGE_CACHE: Path = DEFAULT_DATA_DIR / "images"

# --------------------------------------------------------------------------- #
# Model / API defaults (per-provider endpoints and default models: PROVIDERS)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL: str = "qwen-vl-max-latest"
DEFAULT_TEMPERATURE: float = 0.2
# Long reasoning traces (esp. thinking models) overflow 4096 and get cut off
# mid-JSON, which no parser can recover. 8192 gives comfortable headroom.
DEFAULT_MAX_TOKENS: int = 8192
# When an output is still truncated (finish_reason == "length"), retry once with
# max_tokens bumped up to this ceiling before giving up and flagging it.
TRUNCATION_RETRY_MAX_TOKENS: int = 16384
DEFAULT_TIMEOUT: int = 120
DEFAULT_RETRY: int = 3
DEFAULT_SLEEP: float = 0.5

# DashScope OpenAI-compatible endpoint (used by the reference implementation).
DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Both providers speak the OpenAI chat API, so only the endpoint, the key and
# the model naming differ. ``--provider auto`` picks the first provider, in
# this order, whose key is set.
PROVIDERS: dict[str, dict] = {
    "dashscope": {
        "base_url": DASHSCOPE_BASE_URL,
        "key_env": ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        "default_model": DEFAULT_MODEL,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": ("OPENROUTER_API_KEY",),
        "default_model": "qwen/qwen3-vl-235b-a22b-instruct",
    },
}
PROVIDER_CHOICES: tuple[str, ...] = ("auto", *PROVIDERS)

# --------------------------------------------------------------------------- #
# Sampling defaults
# --------------------------------------------------------------------------- #
DEFAULT_SEED: int = 42
# None = keep the whole pool. Benchmark mode also retains multi-image tasks.
DEFAULT_SAMPLE_SIZE: int | None = None

# Placeholder text written into raw when an image byte blob is stripped out.
IMAGE_BYTES_PLACEHOLDER: str = "<image_bytes_removed>"
