import asyncio
import uuid
import weakref

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


def _mean_target_logprob(entries, target_ids):
    """Mean target log-prob."""
    tokens = [token for _, token in entries]
    width = len(target_ids)
    for k in range(len(tokens) - width, -1, -1):
        if tokens[k : k + width] == target_ids:
            window = [lp for lp, _ in entries[k : k + width]]
            if any(lp is None for lp in window):
                return None
            return sum(window) / width
    return None


class VerlEngine(RolloutEngine):
    supports_recoil = True  # accepts routing_key and provides score_target

    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=rollout_manager.server_handles)
        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        routing_key = kwargs.pop("routing_key", None)  # sticky server for prefix-cache reuse; rids stay unique

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
        if self.config.actor_rollout_ref.rollout.name == "sglang":  # stop at the per-call budget, not the trajectory budget
            sampling_params["max_new_tokens"] = int(max_tokens)

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]

        if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:

            def process():
                images = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
                return images, self.processor(text=[prompt], images=images)

            image_data, model_inputs = await asyncio.to_thread(process)
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)
        else:
            image_data = None
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        token_output: TokenOutput = await self.server_manager.generate(request_id=application_id, prompt_ids=request_prompt_ids, image_data=image_data, sampling_params=sampling_params, routing_key=routing_key)  # type: ignore
        completion_ids: list[int] = token_output.token_ids

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=[],
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    def _probe_images(self, messages):
        """Cached probe images."""
        from rllm.parser.mm_cumulative import fetch, pixel_inputs

        cache = self.__dict__.setdefault("_image_cache", {})
        images = [im for m in messages if m["role"] == "user" for im in (m.get("images") or []) if im is not None]
        fetched, counts = [], []
        for image in images:
            key = id(image)
            entry = cache.get(key)
            if entry is None or entry[0]() is not image:
                resized = fetch(self.processor, image)
                entry = (weakref.ref(image, lambda _, k=key: cache.pop(k, None)), resized, pixel_inputs(self.processor, resized)[1])
                cache[key] = entry
            fetched.append(entry[1])
            counts.append(entry[2])
        return fetched, counts

    async def score_target(self, messages: list[dict], prefix_text: str, target_text: str, *, suffix_text: str = "", routing_key=None, max_target_tokens: int = 32) -> float | None:
        """Prefill log-prob of a target."""

        def prepare():
            prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=self.accumulate_reasoning) + prefix_text
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)[:max_target_tokens]
            suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False) if suffix_text else []
            images, counts = self._probe_images(messages) if self.processor is not None else ([], [])
            return prompt_ids, target_ids, suffix_ids, images or None, sum(c - 1 for c in counts)

        prompt_ids, target_ids, suffix_ids, images, expansion = await asyncio.to_thread(prepare)
        if not target_ids:
            return None
        # Suffix cannot affect preceding target probabilities in a causal model.
        # The server scores one specified next token at a time; image expansion
        # is handled by SGLang, never by guessed input-logprob offsets.
        entries = await self.server_manager.score(
            request_id=str(uuid.uuid4()), prompt_ids=prompt_ids + target_ids,
            image_data=images, logprob_start_len=len(prompt_ids) - 1,
            routing_key=routing_key,
        )
        value = _mean_target_logprob(entries or [], target_ids)
        if value is None:
            raise RuntimeError("RECOIL target scoring returned no matching probabilities")
        return value

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])

    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])
