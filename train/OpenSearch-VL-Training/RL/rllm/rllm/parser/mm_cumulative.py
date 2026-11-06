"""Multimodal cumulative tokenization."""
from __future__ import annotations

from dataclasses import dataclass

import torch


class PromptTooLong(ValueError):
    pass


@dataclass
class CumulativeTokens:
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    turn_ids: torch.Tensor
    multi_modal_inputs: dict
    num_turns: int
    dropped_messages: int


def fetch(processor, image):
    """Model-view image."""
    from qwen_vl_utils import fetch_image

    return fetch_image(image if isinstance(image, dict) else {"image": image}, image_patch_size=processor.image_processor.patch_size)


def pixel_inputs(processor, fetched):
    """Pixel inputs and token count."""
    inputs = processor.image_processor(images=[fetched], return_tensors="pt")
    return inputs, int(inputs["image_grid_thw"][0].prod().item()) // processor.image_processor.merge_size**2


def process_image(processor, image, cache: dict | None = None):
    entry = cache.get(id(image)) if cache is not None else None
    if entry is not None and entry[0] is image:
        return entry[1], entry[2]
    inputs, count = pixel_inputs(processor, fetch(processor, image))
    if cache is not None:
        cache[id(image)] = (image, inputs, count)
    return inputs, count


def _encode(parser, text, message_list, cache):
    ids = parser.tokenizer.encode(text, add_special_tokens=False)
    processor = getattr(parser, "processor", None)
    images = [im for m in message_list for im in (m.get("images") or []) if im is not None]
    if processor is None or not images:
        return ids, []
    processed = [process_image(processor, image, cache) for image in images]
    pad = parser.tokenizer.convert_tokens_to_ids(getattr(parser, "image_token", "<|image_pad|>"))
    expanded, k = [], 0
    for token in ids:
        if token == pad:
            if k >= len(processed):
                raise ValueError("More image placeholders than images in a message")
            expanded.extend([pad] * processed[k][1])
            k += 1
        else:
            expanded.append(token)
    if k != len(processed):
        raise ValueError("Image placeholder count does not match the attached images")
    return expanded, [inputs for inputs, _ in processed]


def tokenize_cumulative_mm(parser, messages, *, max_prompt_length=None, max_response_length=None, cache=None) -> CumulativeTokens:
    """Tokenize a multimodal trajectory."""
    try:
        first = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    except StopIteration:
        raise ValueError("No assistant message found in chat_completions") from None

    prompt_text = parser.parse(messages[:first], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
    prompt_ids, prompt_images = _encode(parser, prompt_text, messages[:first], cache)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        raise PromptTooLong(f"prompt has {len(prompt_ids)} tokens > {max_prompt_length}")

    response_ids, response_mask, turn_ids, response_images = [], [], [], []
    turn, dropped = -1, 0
    for index in range(first, len(messages)):
        message = messages[index]
        if message["role"] == "assistant":
            text = parser.parse([message], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
            ids, images = parser.tokenizer.encode(text[len(parser.generation_prompt):], add_special_tokens=False), []
            mask_value, turn_value = 1, turn + 1
        else:
            text = parser.parse([message], is_first_msg=False, add_generation_prompt=True, accumulate_reasoning=False)
            ids, images = _encode(parser, text, [message], cache)
            mask_value, turn_value = 0, -1
        if max_response_length is not None and len(response_ids) + len(ids) > max_response_length:
            dropped = len(messages) - index
            break
        turn = max(turn, turn_value)
        response_ids.extend(ids)
        response_mask.extend([mask_value] * len(ids))
        turn_ids.extend([turn_value] * len(ids))
        response_images.extend(images)

    processed = prompt_images + response_images
    multi_modal_inputs = {}
    if processed:
        multi_modal_inputs = {key: torch.cat([p[key] for p in processed], dim=0) for key in ("pixel_values", "image_grid_thw")}
    return CumulativeTokens(
        prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
        response_ids=torch.tensor(response_ids, dtype=torch.long),
        response_mask=torch.tensor(response_mask, dtype=torch.long),
        turn_ids=torch.tensor(turn_ids, dtype=torch.long),
        multi_modal_inputs=multi_modal_inputs,
        num_turns=turn + 1,
        dropped_messages=dropped,
    )
