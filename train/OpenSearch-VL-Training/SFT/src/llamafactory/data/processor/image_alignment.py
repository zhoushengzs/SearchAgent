"""Qwen image-only SFT: truncate token spans and their image list together."""


def image_spans(tokens, start, pad, end):
    spans = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == start:
            j = i + 1
            while j < len(tokens) and tokens[j] == pad:
                j += 1
            if j == i + 1 or j >= len(tokens) or tokens[j] != end:
                raise ValueError("Malformed Qwen image span before truncation; refusing to guess image mapping.")
            spans.append((i, j + 1))
            i = j + 1
        elif token in (pad, end):
            raise ValueError("Unpaired Qwen vision token before truncation.")
        else:
            i += 1
    return spans


def safe_prefix(length, spans):
    for start, end in spans:
        if start < length < end:
            return start
    return length


def encode_image_example(owner, prompt, response, system, tools, images, videos, audios):
    """Same last-turn supervision as upstream; images is a PRIVATE mutable copy."""
    from .processor_utils import infer_seqlen
    from ...extras.constants import IGNORE_INDEX
    plugin = owner.template.mm_plugin
    tokenizer = owner.tokenizer
    if videos or audios:
        raise ValueError("Image alignment patch supports image-only samples, not mixed video/audio.")
    special = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
    ids = [tokenizer.convert_tokens_to_ids(t) for t in special]
    if len(set(ids)) != 3 or any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError("Qwen vision special tokens not found in tokenizer.")
    messages = plugin.process_messages(prompt + response, images, videos, audios, owner.processor)
    initial, initial_labels = plugin.process_token_ids([], [], images, videos, audios, tokenizer, owner.processor)
    if initial or initial_labels:
        raise ValueError("Unexpected Qwen token prefix; refusing an unsupported multimodal layout.")
    pairs = list(owner.template.encode_multiturn(tokenizer, messages, system, tools))
    indexed = []
    image_index = 0
    for source, target in pairs:
        source_spans = image_spans(source, *ids)
        if image_spans(target, *ids):
            raise ValueError("Images in assistant targets are not supported by this patch.")
        indexed.append((source, target, source_spans, image_index))
        image_index += len(source_spans)
    if image_index != len(images):
        raise ValueError(f"Before truncation: {image_index} encoded image spans vs {len(images)} images.")
    if owner.data_args.mask_history:
        indexed.reverse()
    input_ids, labels, kept = [], [], []
    total = 1 if owner.template.efficient_eos else 0
    for turn_idx, (source, target, spans, first_image) in enumerate(indexed):
        if total >= owner.data_args.cutoff_len:
            break
        source_len, target_len = infer_seqlen(len(source), len(target), owner.data_args.cutoff_len - total)
        source_len = safe_prefix(source_len, spans)
        source = source[:source_len]
        target = target[:target_len]
        total += len(source) + len(target)
        selected = [first_image + j for j, (_, end) in enumerate(spans) if end <= source_len]
        if owner.data_args.train_on_prompt:
            source_labels = source[:]
        elif owner.template.efficient_eos and turn_idx != 0 and source:
            source_labels = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source) - 1)
        else:
            source_labels = [IGNORE_INDEX] * len(source)
        target_labels = ([IGNORE_INDEX] * len(target)
                         if owner.data_args.mask_history and turn_idx != 0 else target[:])
        if owner.data_args.mask_history:
            input_ids = source + target + input_ids
            labels = source_labels + target_labels + labels
            kept = selected + kept
        else:
            input_ids += source + target
            labels += source_labels + target_labels
            kept += selected
    if owner.template.efficient_eos:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    if len(input_ids) != len(labels) or len(input_ids) > owner.data_args.cutoff_len:
        raise ValueError("Invalid sequence/label length after image-aware truncation.")
    if len(image_spans(input_ids, *ids)) != len(kept):
        raise ValueError("Image spans and retained image indices disagree after truncation.")
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("No supervised tokens remain after truncation.")
    original_count = len(images)
    images[:] = [images[index] for index in kept]
    if original_count != len(images):
        from ...extras import logging
        logging.get_logger(__name__).warning_rank0(
            f"Image-aware truncation: kept original image indices {kept} out of {original_count}; "
            "removed images only where the corresponding historical token spans were removed."
        )
    return input_ids, labels
