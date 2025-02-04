import time
import copy
from typing import List, Optional, Callable, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .sample_utils import make_sampler
from mlx_lm.utils import wired_limit, cache, maybe_quantize_kv_cache, GenerationResponse
from mlx_lm import load, generate, stream_generate


def prompt_to_cache(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt_ids: List[int],
    c: Optional[List[cache.KVCache]] = None,
    prefill_step_size: int = 512,
) -> tuple[List[cache.KVCache], int]:
    """
    Process a prompt and fill the KV cache.

    Args:
        model (nn.Module): The language model.
        tokenizer (PreTrainedTokenizer or TokenizerWrapper): The tokenizer.
        prompt_ids (List[int]): List of token IDs for the prompt.
        c (Optional[List[cache.KVCache]]): The KV cache to fill. If None, a new cache is created.
        prefill_step_size (int): Step size used when processing the prompt.

    Returns:
        Tuple[List[cache.KVCache], int]: The filled KV cache and total prompt length.
    """
    prompt_ids = mx.array(prompt_ids)
    if c is None:
        c = cache.make_prompt_cache(model)
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    total_prompt_len = prompt_ids.shape[0]
    processed = 0
    while processed < total_prompt_len:
        chunk_end = min(processed + prefill_step_size, total_prompt_len)
        inputs_chunk = prompt_ids[processed:chunk_end]
        _ = model(inputs_chunk[None], cache=c)
        processed = chunk_end

    return c, total_prompt_len


def _prefill_cache(
    model: nn.Module,
    prompt_ids: mx.array,
    prompt_cache: List[cache.KVCache],
    generation_stream: mx.Stream,
    prefill_step_size: int,
) -> tuple[int, float]:
    """
    Prefill the prompt cache by running the prompt through the model in chunks.
    
    Returns:
        total_prompt_len: number of tokens in the prompt.
        prompt_tps: prompt tokens per second (for logging purposes).
    """
    total_prompt_len = prompt_ids.shape[0]
    processed = 0
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        while processed < total_prompt_len:
            chunk_end = min(processed + prefill_step_size, total_prompt_len)
            inputs_chunk = prompt_ids[processed:chunk_end]
            with mx.stream(generation_stream):
                _ = model(inputs_chunk[None], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
            processed = chunk_end
            mx.metal.clear_cache()
        prompt_time = time.perf_counter() - tic
    prompt_tps = (total_prompt_len * 1) / prompt_time if prompt_time > 0 else 0.0  # Only one prompt sequence.
    return total_prompt_len, prompt_tps


def _replicate_cache(
    prompt_cache: List[cache.KVCache], batch_size: int
) -> List[cache.KVCache]:
    """
    Replicate the prompt cache along the batch dimension.
    """
    prompt_cache = copy.copy(prompt_cache)
    for c in prompt_cache:
        c.keys = mx.repeat(c.keys, repeats=batch_size, axis=0)
        c.values = mx.repeat(c.values, repeats=batch_size, axis=0)
    return prompt_cache


def generate_batched(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: List[int],
    batch_size: int,
    *,
    prompt_cache: Optional[List[cache.KVCache]] = None,
    verbose: bool = False,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    repetition_penalty: float = 1.0,
    repetition_context_size: int = 20,
    prefill_step_size: int = 512,
    **kwargs,
) -> tuple[List[str], List[cache.KVCache], int, List[int], List[bool]]:
    """
    Generate multiple responses in parallel from the same prompt.
    """
    # Ensure tokenizer is wrapped.
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Use the last token of the prompt as the starting input.
    prompt_ids = mx.array([prompt[-1]])

    # Initialize decoded texts and ended flags.
    decoded_texts = ["" for _ in range(batch_size)]
    ended = [False] * batch_size
    sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)

    generation_stream = mx.new_stream(mx.default_device())
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model)
        with wired_limit(model, [generation_stream]):
            total_prompt_len, prompt_tps = _prefill_cache(
                model, prompt_ids, prompt_cache, generation_stream, prefill_step_size
            )
    else:
        total_prompt_len = prompt_cache[0].offset  # Assumes the offset is stored.

    # Replicate cache and starting token across batch.
    prompt_cache = _replicate_cache(prompt_cache, batch_size)
    y = mx.repeat(prompt_ids[-1:][None, :], repeats=batch_size, axis=0)
    tokens_so_far = [[] for _ in range(batch_size)]

    tic = time.perf_counter()
    n = 0
    while n < max_tokens:
        with mx.stream(generation_stream):
            logits = model(y, cache=prompt_cache)
            logits = logits[:, -1, :]  # Use only the last token’s logits.
            mx.async_eval(logits)

        # Compute log-probabilities and sample next tokens.
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampled_tokens = sampler(logprobs).tolist()
        next_tokens_list = []
        for i in range(batch_size):
            if ended[i]:
                # For ended sequences, re-use a token so that shape stays consistent.
                next_tokens_list.append(y[i, 0])
                continue
            token = sampled_tokens[i]
            if token in tokenizer.eos_token_ids:
                ended[i] = True
            next_tokens_list.append(token)
            if not ended[i]:
                tokens_so_far[i].append(token)

        y = mx.array(next_tokens_list).reshape(batch_size, 1)
        n += 1

        if all(ended):
            break

        if n % 256 == 0:
            mx.metal.clear_cache()

    generation_time = time.perf_counter() - tic
    total_generated_tokens = sum(len(seq) for seq in tokens_so_far)
    for i in range(batch_size):
        decoded_texts[i] = tokenizer.decode(tokens_so_far[i])

    if verbose:
        for i, txt in enumerate(decoded_texts):
            print("=" * 10)
            print(f"Batch {i}: {txt}")
        print("=" * 10)
        if not decoded_texts:
            print("No text generated for this prompt.")
        else:
            print(
                f"Prompt tokens (per sequence): {total_prompt_len}, "
                f"Prompt TPS (across all sequences): {prompt_tps:.3f}"
            )
            print(
                f"Generation tokens (max per sequence): {n}, "
                f"Generation TPS (across all sequences): "
                f"{(total_generated_tokens)/(generation_time+1e-9):.3f}"
            )
            peak_mem = mx.metal.get_peak_memory() / 1e9
            print(f"Peak memory: {peak_mem:.3f} GB")

    mx.metal.clear_cache()
    return decoded_texts, prompt_cache, total_prompt_len, [len(x) for x in tokens_so_far], ended


def generate_batched_stream(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: List[int],
    batch_size: int,
    *,
    prompt_cache: Optional[List[cache.KVCache]] = None,
    verbose: bool = False,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    repetition_penalty: float = 1.0,
    repetition_context_size: int = 20,
    prefill_step_size: int = 512,
    **kwargs,
):
    """
    Generate multiple responses in parallel from the same prompt, yielding updates as tokens are generated.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_ids = mx.array([prompt[-1]])
    decoded_texts = ["" for _ in range(batch_size)]
    ended = [False] * batch_size
    sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)

    generation_stream = mx.new_stream(mx.default_device())
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model)
        with wired_limit(model, [generation_stream]):
            total_prompt_len, _ = _prefill_cache(
                model, prompt_ids, prompt_cache, generation_stream, prefill_step_size
            )
    else:
        total_prompt_len = prompt_cache[0].offset

    prompt_cache = _replicate_cache(prompt_cache, batch_size)
    y = mx.repeat(prompt_ids[-1:][None, :], repeats=batch_size, axis=0)
    tokens_so_far = [[] for _ in range(batch_size)]
    n = 0

    with wired_limit(model, [generation_stream]):
        while n < max_tokens:
            logits = model(y, cache=prompt_cache)
            logits = logits[:, -1, :]
            mx.async_eval(logits)
            sampled_tokens = sampler(logits).tolist()
            for i in range(batch_size):
                token = sampled_tokens[i]
                if token in tokenizer.eos_token_ids:
                    ended[i] = True
                if not ended[i]:
                    tokens_so_far[i].append(token)
            y = mx.array(sampled_tokens).reshape(batch_size, 1)
            n += 1
            decoded_texts = [tokenizer.decode(seq) for seq in tokens_so_far]
            yield {"type": "update", "decoded_texts": decoded_texts}
            if all(ended):
                break

    yield {
        "type": "final",
        "decoded_texts": [tokenizer.decode(seq) for seq in tokens_so_far],
        "generated_lengths": [len(seq) for seq in tokens_so_far],
        "total_prompt_len": total_prompt_len,
        "ended": ended,
        "prompt_cache": prompt_cache,
    }
