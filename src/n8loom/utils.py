import time
import copy
from typing import List, Optional, Callable, Union, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .sample_utils import make_sampler
from .models.llama import Model
from mlx_lm.utils import wired_limit, cache, maybe_quantize_kv_cache, GenerationResponse, get_model_path, load_model, load_adapters, load_tokenizer
from mlx_lm import load, generate, stream_generate
import importlib
import math
MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
}

def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"n8loom.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs

def load_for_loom(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If ``False`` eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(model_path, lazy, get_model_classes=_get_classes)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(
        model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id", None)
    )

    return model, tokenizer

def prompt_to_cache(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt_ids: List[int],
    c: Optional[List[cache.KVCache]] = None,
    prefill_step_size: int = 512,
    offset: int = 0,
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

    total_prompt_len = prompt_ids.shape[0] - offset
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
    stop_strings: List[str] = [],
    **kwargs,
) -> Tuple[
    List[List[int]],        # Generated token sequences
    List[cache.KVCache],   # Generation cache
    int,                   # total_prompt_len
    List[int],             # lengths of the generated token sequences
    List[bool]             # ended flags
]:
    """
    Generate multiple responses in parallel from the same prompt, returning token sequences.
    
    Returns:
        - generated_token_seqs: list of token sequences (one per batch element).
        - cache_gen: the per-token generation cache (list of KVCache).
        - total_prompt_len: number of prompt tokens used.
        - generated_lengths: lengths of the token sequences generated for each batch element.
        - ended: boolean flags indicating whether each sequence ended via an EOS token.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Prompt to feed into the model is the last token (non-traditional attention pattern).
    prompt_ids = mx.array([prompt[-1]])

    ended = [False] * batch_size
    ended_due_to_eos = [False] * batch_size
    sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)

    generation_stream = mx.new_stream(mx.default_device())
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model)
        with wired_limit(model, [generation_stream]):
            total_prompt_len, prompt_tps = _prefill_cache(
                model, prompt_ids, prompt_cache, generation_stream, prefill_step_size
            )
    else:
        total_prompt_len = prompt_cache[0].offset

    # Create a separate generation cache for newly generated tokens.
    cache_gen = cache.make_prompt_cache(model)
    cache_step = max(2 ** math.floor(math.log2( 256 / batch_size)), 4)
    for c in cache_gen:
        c.step = cache_step

    # The initial (last) token is repeated `batch_size` times
    y = mx.repeat(prompt_ids[-1:][None, :], repeats=batch_size, axis=0)

    # We accumulate generated tokens for each batch element
    tokens_so_far = [[] for _ in range(batch_size)]
    stop_string_tok_lengths = [len(tokenizer.encode(s)) for s in stop_strings]
    tic = time.perf_counter()
    n = 0
    has_stop_strings = len(stop_strings) > 0
    with wired_limit(model, [generation_stream]):
        while n < max_tokens:
            logits = model(y, cache=prompt_cache, cache_gen=cache_gen)
            logits = logits[:, -1, :]  # only the logits for the new position
            mx.async_eval(logits)

            # Compute probabilities and sample new tokens
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled_tokens = sampler(logprobs).tolist()

            next_tokens_list = []
            for i in range(batch_size):
                if ended[i]:
                    # If sequence already ended, repeat the previous token
                    next_tokens_list.append(y[i, 0])
                    continue

                token = sampled_tokens[i]
                if token in tokenizer.eos_token_ids:
                    ended[i] = True
                    ended_due_to_eos[i] = True
                # Check for stop strings
                if has_stop_strings:
                    for j, stop_string in enumerate(stop_strings):
                        max_length_to_check = 2 * stop_string_tok_lengths[j]
                        last_toks = tokens_so_far[i][-max_length_to_check:]
                        last_toks_str = tokenizer.decode(last_toks)
                        if stop_string in last_toks_str:
                            ended[i] = True
                next_tokens_list.append(token)
                if not ended[i]:
                    tokens_so_far[i].append(token)

            y = mx.array(next_tokens_list).reshape(batch_size, 1)
            n += 1
            if n % cache_step == 0:
                mx.metal.clear_cache()
            if all(ended):
                break


    generation_time = time.perf_counter() - tic
    total_generated_tokens = sum(len(seq) for seq in tokens_so_far)

    if verbose:
        for i, seq in enumerate(tokens_so_far):
            print("=" * 10)
            print(f"Batch {i} tokens:", seq)
            print("Decoded text:", tokenizer.decode(seq))
        print("=" * 10)
        if not tokens_so_far:
            print("No tokens generated for this prompt.")
        else:
            print(
                f"Prompt tokens (per sequence): {total_prompt_len}"
            )
            print(
                f"Generation tokens (max per sequence): {n}, "
                f"Overall generation TPS: "
                f"{(total_generated_tokens)/(generation_time+1e-9):.3f}"
            )
            peak_mem = mx.metal.get_peak_memory() / 1e9
            print(f"Peak memory: {peak_mem:.3f} GB")

    mx.metal.clear_cache()

    # Return the token sequences instead of decoded strings
    return (
        tokens_so_far,
        cache_gen,
        total_prompt_len,
        [len(seq) for seq in tokens_so_far],
        ended_due_to_eos,
    )
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
    stop_strings: List[str] = [],  # <-- Added stop_strings parameter.
    **kwargs,
):
    """
    Generate multiple responses in parallel from the same prompt, yielding updates as tokens are generated.
    Returns tokens (rather than decoded strings) in each update, but also includes a decoded text if desired.

    Now supports stop strings: if any decoded text contains one of the stop strings (within a window of
    twice the stop string token length), that sequence is ended.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Use the last token of the prompt as the starting token.
    prompt_ids = mx.array([prompt[-1]])
    ended = [False] * batch_size
    ended_due_to_eos = [False] * batch_size
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

    # Create a separate generation cache for new tokens.
    cache_gen = cache.make_prompt_cache(model)
    cache_step = max(2 ** math.floor(math.log2(256 / batch_size)), 4)
    for c in cache_gen:
        c.step = cache_step

    # Repeat the starting token for each batch element.
    y = mx.repeat(prompt_ids[-1:][None, :], repeats=batch_size, axis=0)
    tokens_so_far = [[] for _ in range(batch_size)]

    # Pre-compute token lengths for each stop string.
    stop_string_tok_lengths = [len(tokenizer.encode(s)) for s in stop_strings]
    has_stop_strings = len(stop_strings) > 0

    n = 0
    with wired_limit(model, [generation_stream]):
        while n < max_tokens:
            logits = model(y, cache=prompt_cache, cache_gen=cache_gen)
            logits = logits[:, -1, :]  # Only the logits for the new position.
            mx.async_eval(logits)
            sampled_tokens = sampler(logits).tolist()

            new_tokens = []
            for i in range(batch_size):
                if ended[i]:
                    # If this sequence has already ended, repeat its last token.
                    new_tokens.append(y[i, 0])
                    continue

                token = sampled_tokens[i]
                if token in tokenizer.eos_token_ids:
                    ended[i] = True
                    ended_due_to_eos[i] = True

                # Check if any stop string is found in a window of the generated tokens.
                if has_stop_strings:
                    for j, stop_string in enumerate(stop_strings):
                        max_length_to_check = 2 * stop_string_tok_lengths[j]
                        # Consider the last tokens generated so far.
                        last_toks = tokens_so_far[i][-max_length_to_check:]
                        last_toks_str = tokenizer.decode(last_toks)
                        if stop_string in last_toks_str:
                            ended[i] = True
                new_tokens.append(token)
                if not ended[i]:
                    tokens_so_far[i].append(token)

            y = mx.array(new_tokens).reshape(batch_size, 1)
            n += 1
            if n % cache_step == 0:
                mx.metal.clear_cache()
            # Yield partial update with tokens and decoded text.
            yield {
                "type": "update",
                "tokens": [ts[:] for ts in tokens_so_far],  # copy current tokens
                "decoded_texts": [tokenizer.decode(ts) for ts in tokens_so_far],
                "ended": ended[:],
            }
            if all(ended):
                break

    mx.metal.clear_cache()
    # Final yield with all tokens and additional generation metadata.
    yield {
        "type": "final",
        "tokens": tokens_so_far,
        "decoded_texts": [tokenizer.decode(seq) for seq in tokens_so_far],
        "generated_lengths": [len(seq) for seq in tokens_so_far],
        "total_prompt_len": total_prompt_len,
        "ended": ended_due_to_eos,
        "prompt_cache": cache_gen,
    }
