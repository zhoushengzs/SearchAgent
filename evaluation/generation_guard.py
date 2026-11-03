"""Conservative, opt-in repetition detection for bounded diagnostic inference."""
from collections import defaultdict


def repeated_token_span(tokens, span=32, repetitions=3):
    """Find a long exact token phrase occurring repeatedly without overlap."""
    positions = defaultdict(list)
    for index in range(len(tokens) - span + 1):
        key = tuple(tokens[index:index + span])
        seen = positions[key]
        if not seen or index - seen[-1] >= span:
            seen.append(index)
            if len(seen) >= repetitions:
                return {'span_tokens': span, 'repetitions': repetitions,
                        'positions': seen[-repetitions:]}
    return None


def stopping_guard(prompt_tokens):
    from transformers import StoppingCriteria

    class RepetitionGuard(StoppingCriteria):
        hit = None

        def __call__(self, input_ids, scores, **kwargs):
            count = input_ids.shape[1] - prompt_tokens
            if count >= 96 and count % 16 == 0:
                self.hit = repeated_token_span(input_ids[0, -min(count, 1536):].tolist())
            return self.hit is not None

    return RepetitionGuard()
