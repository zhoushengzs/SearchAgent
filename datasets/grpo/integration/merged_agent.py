from __future__ import annotations

import asyncio
import io
import os
import time
from datetime import datetime
from typing import Any, List, Optional, TYPE_CHECKING

import requests
from PIL import Image
import re
from collections import Counter

# rLLM imports
if TYPE_CHECKING:
    from rllm.engine.rollout import RolloutEngine

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 50
MAX_PROMPT_LENGTH_PER_RUN = 64000
MAX_RESPONSE_LENGTH_PER_RUN = 4096

from project_runtime.session import get_all_tools, system_prompt

DEEPRESEARCH_SYSTEM_PROMPT = system_prompt()


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


def analyze_repetition_ngram(text: str, n: int = 30, threshold: float = 0.5):
    """
    Use N-grams to detect repetition in a string.

    Args:
        text (str): Input text to analyze.
        n (int): N-gram window size (default 10).
            - For long repetitive sequences, 10-20 is recommended.
        threshold (float): Distinct-N threshold (0~1).
            - Values below this indicate heavy repetition (default 0.5).

    Returns:
        bool: True if repetition is detected, False otherwise.
    """
    if not text or len(text) < n:
        print("text is too short, cannot analyze.")
        return False

    # 1. Generate N-grams (character-level sliding window).
    # List comprehension: slice from index i to i+n.
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]

    total_count = len(ngrams)
    if total_count == 0:
        return False

    # 2. Count frequencies.
    ngram_counts = Counter(ngrams)
    unique_count = len(ngram_counts)

    # 3. Compute Distinct-N (unique count / total count).
    # Repetitive text is typically < 0.4.
    distinct_ratio = unique_count / total_count

    # 4. Determine repetition.
    is_repetitive = distinct_ratio < threshold

    return is_repetitive


def count_words(text: str) -> int:
    # Match segments that look like English words.
    # Rule: starts and ends with a letter, may contain letters, apostrophes, or hyphens.
    pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    words = pattern.findall(text)
    return len(words)


def build_text_completion_prompt(
    messages: list[dict], allow_special: bool = True
) -> str:
    """
    Build text completion prompt from messages list.
    Adapted from qwen_agent.utils.utils.build_text_completion_prompt

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        allow_special: Whether to allow special tokens (for compatibility)

    Returns:
        Formatted prompt string
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_parts = []

    # Handle system message
    if messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        prompt_parts.append(f"{im_start}system\n{sys_content}{im_end}")
        messages = messages[1:]

    # Ensure chat completes with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages + [{"role": "assistant", "content": ""}]

    # Format each message
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"{im_start}{role}\n{content}{im_end}")

    return "\n".join(prompt_parts)


# Safety cap on how many freshly produced images we forward back to the model
# per tool call. Visual tools currently emit at most one new image per call, but
# the cap protects context length and vision-encoder memory in case future tools
# emit several at once.
_MAX_ATTACHED_IMAGES_PER_TOOL = 4

# Best-effort timeout for downloading tool-produced images hosted on COS / web.
_IMAGE_DOWNLOAD_TIMEOUT = 30


def _load_pil_from_payload(payload: Any) -> Optional[Image.Image]:
    """Best-effort conversion of a tool-produced image payload into PIL.Image.

    Visual tools store newly produced images in the shared ``image_paths`` dict
    using one of several representations (see ``visual_tools._resolve_image_ref``
    for the full taxonomy):

    * ``http(s)://`` URL  (the default — uploaded to COS)
    * an absolute local path (fallback when COS upload is disabled)
    * raw ``bytes`` containing encoded image data
    * a ``PIL.Image.Image`` instance
    * a HuggingFace-style ``dict`` with ``bytes`` / ``path`` / ``url`` keys

    Returns ``None`` if the payload cannot be decoded; callers should silently
    skip such entries so a single corrupted image never aborts the rollout.
    """
    if isinstance(payload, Image.Image):
        return payload

    if isinstance(payload, bytes):
        try:
            return Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    if isinstance(payload, str):
        if payload.startswith(("http://", "https://")):
            try:
                resp = requests.get(payload, timeout=_IMAGE_DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        if payload.startswith("data:image/"):
            try:
                import base64

                _, encoded = payload.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        if os.path.exists(payload):
            try:
                return Image.open(payload).convert("RGB")
            except Exception:  # noqa: BLE001
                return None

    if isinstance(payload, dict):
        for key in ("bytes", "path", "url", "data"):
            val = payload.get(key)
            if val is None:
                continue
            pil = _load_pil_from_payload(val)
            if pil is not None:
                return pil

    return None


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        default_max_tries: int = 3,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
            system_prompt: Optional custom system prompt
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt
        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.default_max_tries = default_max_tries

        # Smart context management using actual API consumption
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.max_prompt_tokens = MAX_PROMPT_LENGTH_PER_RUN
        self.max_response_tokens = MAX_RESPONSE_LENGTH_PER_RUN

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

    async def call_server(
        self, messages: list[dict], max_tries: Optional[int] = None, **kwargs
    ):
        """Call rollout engine once; assumes XML ReAct format."""
        try:
            # Force per-round limits from DeepResearchAgent without local token estimation.
            if hasattr(self.rollout_engine, "max_prompt_length"):
                self.rollout_engine.max_prompt_length = int(self.max_prompt_tokens)
            if hasattr(self.rollout_engine, "max_response_length"):
                self.rollout_engine.max_response_length = int(self.max_response_tokens)

            kwargs.pop("max_new_tokens", None)
            kwargs["max_tokens"] = int(self.max_response_tokens)
            response = await self.rollout_engine.get_model_response(
                messages=messages, **kwargs
            )

            if hasattr(response, "prompt_length") and hasattr(
                response, "completion_length"
            ):
                self.total_prompt_tokens += response.prompt_length
                self.total_completion_tokens += response.completion_length

            return response
        except Exception as exc:  # noqa: BLE001
            print(f"call_server failed: {exc}")
            raise

    def record_token_usage(self, response) -> None:
        """Record the latest prompt/completion token count from rollout engine."""
        prompt_tokens = getattr(response, "prompt_length", None)
        completion_tokens = getattr(response, "completion_length", None)

        if prompt_tokens is not None:
            try:
                self.total_prompt_tokens = int(prompt_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_prompt_tokens = 0

        if completion_tokens is not None:
            try:
                self.total_completion_tokens = int(completion_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_completion_tokens = 0

    def get_total_tokens_used(self) -> int:
        """Return the latest prompt + completion token usage reported by the engine."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate prompt length for the next call using the rollout engine's tokenizer."""
        tokenizer = getattr(self.rollout_engine, "tokenizer", None)
        chat_parser = getattr(self.rollout_engine, "chat_parser", None)

        if tokenizer is None or chat_parser is None:
            return self.total_prompt_tokens

        try:
            prompt = chat_parser.parse(
                messages,
                add_generation_prompt=True,
                is_first_msg=True,
                tools=[],
                accumulate_reasoning=getattr(
                    self.rollout_engine, "accumulate_reasoning", False
                ),
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return len(token_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"[TokenEstimator] Failed to estimate prompt tokens: {exc}")
            return self.total_prompt_tokens

    def _build_result(
        self,
        *,
        question: str,
        answer: str | None,
        messages: list[dict],
        prediction: str,
        termination: str,
        rounds: int,
        start_time: float,
        # next_prompt_tokens: int | None = None,
    ) -> dict:
        """Assemble result payload with token usage metadata."""
        token_usage = {
            "prompt": self.total_prompt_tokens,
            "completion": self.total_completion_tokens,
            "max_prompt": self.max_prompt_tokens,
        }

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.time() - start_time,
            "token_usage": token_usage,
        }
        return result

    async def _run(
        self, question: str, answer: str = None, images: list = None,
        image_path: str = None, **kwargs,
    ) -> dict:
        import tempfile
        from project_runtime.session import ProjectSession, parse_action, system_prompt

        start_time = time.time()
        initial_paths = kwargs.pop('initial_image_paths', None)
        kwargs.pop('intermediate_dir', None)
        source_images = images if images is not None else (initial_paths or ([image_path] if image_path else []))
        if set(self.tools) != set(get_all_tools()):
            raise ValueError('The project workflow requires the six SFT evidence tools.')
        with tempfile.TemporaryDirectory(prefix='grpo-evidence-') as directory:
            session = ProjectSession(question, source_images, directory, self.max_llm_calls)
            messages = [{'role': 'system', 'content': self.system_prompt or system_prompt()},
                        session.initial_message()]
            prediction, termination, bad_steps, round = '', 'max_rounds_reached_no_answer', 0, 0
            for round in range(1, self.max_llm_calls + 1):
                try:
                    response = await self.call_server(messages, **kwargs)
                except Exception:
                    termination = 'error'
                    break
                self.record_token_usage(response)
                content = getattr(response, 'text', '') or ''
                assistant = {'role': 'assistant', 'content': content, 'step_error': False}
                messages.append(assistant)
                try:
                    kind, value = parse_action(content)
                except (ValueError, TypeError, KeyError):
                    assistant['step_error'] = True
                    bad_steps += 1
                    result = {'status': 'error', 'error_type': 'InvalidAction',
                              'message': 'Use one complete <think> block and one valid <tool_call> or <response>. '
                                         'Only the six listed vs_ tools are available.'}
                    messages.append(session.message(round + 1, result=result))
                else:
                    if kind == 'answer':
                        prediction, termination = value, 'answer'
                        break
                    try:
                        result, observation = await asyncio.to_thread(
                            session.execute, value['name'], value['arguments'], round)
                    except Exception as exc:
                        result = {'tool': value['name'], 'status': 'error',
                                  'error_type': type(exc).__name__,
                                  'message': 'The requested tool could not execute. Check the supplied arguments.'}
                        observation = session.message(round + 1, result=result)
                    assistant['step_error'] = result.get('status') != 'ok'
                    messages.append(observation)
                    bad_steps = bad_steps + 1 if assistant['step_error'] else 0
                if bad_steps >= 3:
                    termination = 'consecutive_step_errors'
                    break
            return self._build_result(question=question, answer=answer, messages=messages,
                                      prediction=prediction, termination=termination,
                                      rounds=round, start_time=start_time)


    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # Reset token counters for each new task
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)

        Returns:
            Result dictionary
        """
        # Reset token counters for each new run
        self.reset()
        return await self._run(question, answer, images, image_path, **kwargs)


DeepResearchAgent = MultiTurnReactAgent
