import time
from typing import List, Optional, Callable, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import wired_limit, cache, maybe_quantize_kv_cache, GenerationResponse
from mlx_lm import load, generate, stream_generate
import copy

def prompt_to_cache(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt_ids: List[int],
    c: Optional[List[cache.KVCache]] = None,
    prefill_step_size: int = 512,
):
    """
    Process a prompt and fill the KV cache.

    Args:
        model (nn.Module): The language model.
        tokenizer (PreTrainedTokenizer or TokenizerWrapper): The tokenizer.
        prompt (str): The string prompt.
        cache (List[KVCache]): The KV cache to fill. If None, a new cache is created.
        prefill_step_size (int): Step size used when processing the prompt. Default: 512.

    Returns:
        List[KVCache]: The filled KV cache.
    """
    prompt_ids = mx.array(prompt_ids)
    if c is None:
        c = cache.make_prompt_cache(model)
    # Ensure we have a TokenizerWrapper
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Encode the prompt

    # Step 1. "Prefill" / process the prompt in chunks to fill the KV cache
    total_prompt_len = prompt_ids.shape[0]
    processed = 0
    while processed < total_prompt_len:
        chunk_end = min(processed + prefill_step_size, total_prompt_len)
        # Forward pass of shape: (batch_size, chunk_size)
        inputs_chunk = prompt_ids[processed:chunk_end]
        _ = model(inputs_chunk[None], cache=c)
        processed = chunk_end

    return c, total_prompt_len
def generate_batched(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: List[int],
    batch_size: int,
    *,
    prompt_cache: Optional[List[cache.KVCache]] = None,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
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

    Args:
        model (nn.Module): The language model.
        tokenizer (PreTrainedTokenizer or TokenizerWrapper): The tokenizer.
        prompt (str): The string prompt.
        batch_size (int): Number of parallel sequences to generate.
        verbose (bool): If True, prints tokens and timing information. Default: False.
        formatter (Callable): Deprecated. (No longer used)
        max_tokens (int): The maximum number of tokens to generate. Default: 256.
        temp (float): Temperature for sampling. Default: 0.0.
        top_p (float): Nucleus sampling top-p parameter. Default: 0.0.
        min_p (float): Minimum cumulative probability cutoff. Default: 0.0.
        min_tokens_to_keep (int): Ensures a minimum number of tokens remain after filtering. Default: 1.
        repetition_penalty (float): Repetition penalty. Default: 1.0.
        repetition_context_size (int): The context size to consider for the repetition penalty. Default: 20.
        kv_bits (int): Number of bits for KV cache quantization. None = disabled. Default: None.
        kv_group_size (int): Group size for KV cache quantization. Default: 64.
        quantized_kv_start (int): The step to begin using a quantized KV cache. Default: 0.
        max_kv_size (int): The maximum size of the KV cache. Old tokens get overwritten. Default: None.
        prefill_step_size (int): Step size used when processing the prompt (prefill). Default: 512.
        **kwargs: Unused extra kwargs, included for API-compatibility.

    Returns:
        List[str]: A list of decoded text strings of length `batch_size`.
    """
    if formatter is not None:
        print(
            "[Warning] Text formatting is deprecated and no longer used. "
            "The argument will be removed in a future version."
        )
    # Ensure we have a TokenizerWrapper
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Encode the prompt
    prompt_ids = mx.array([prompt[-1]])

    # Prepare to replicate the single prompt for all batch sequences
    # Shape: (batch_size, prompt_length)
    #batched_prompt_ids = mx.repeat(prompt_ids[None, :], repeats=batch_size, axis=0)

    # We'll maintain the partially decoded tokens for each batch element
    # and the final decoded strings
    decoded_texts = ["" for _ in range(batch_size)]

    # Bookkeeping for which sequences have ended
    ended = [False] * batch_size

    # Create any required sampler and logits processors
    sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)
    #logits_processors = make_logits_processors(None, repetition_penalty, repetition_context_size)
    total_prompt_len = -1
    prompt_tps = -1.0
    generation_stream = mx.new_stream(mx.default_device())
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model)
        # Some timers for verbosity
        with wired_limit(model, [generation_stream]):
            tic = time.perf_counter()

            # Step 1. "Prefill" / process the prompt in chunks to fill the KV cache
            total_prompt_len = prompt_ids.shape[0]
            processed = 0
            while processed < total_prompt_len:
                chunk_end = min(processed + prefill_step_size, total_prompt_len)
                # Forward pass of shape: (batch_size, chunk_size)
                inputs_chunk = prompt_ids[processed:chunk_end]
                with mx.stream(generation_stream):
                    _ = model(inputs_chunk[None], cache=prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                processed = chunk_end
                mx.metal.clear_cache()

            # The time spent so far was for prompt processing
            prompt_time = time.perf_counter() - tic
            if total_prompt_len == 0:
                prompt_tps = 0.0
            else:
                prompt_tps = (batch_size * total_prompt_len) / prompt_time
    else:
        total_prompt_len = prompt_cache[0].offset
    prompt_cache = copy.copy(prompt_cache)
    y = mx.repeat(prompt_ids[-1:][None, :], repeats=batch_size, axis=0)
    for c in prompt_cache:
        c.keys = mx.repeat(c.keys, repeats=batch_size, axis=0)
        c.values = mx.repeat(c.values, repeats=batch_size, axis=0)
    # We also keep track of the entire decoded tokens. We start them out with the entire prompt.
    # We'll append new tokens as they are generated.  shape: (B, T_so_far)
    #tokens_so_far = batched_prompt_ids
    tokens_so_far = [[] for _ in range(batch_size)]

    # Step 2. Start generating new tokens (decode) until all ended or max_tokens is reached
    tic = time.perf_counter()
    n = 0
    while True:
        if n >= max_tokens:
            break

        # Forward pass for the current token(s). The model expects shape (B, L).
        # L=1 for incremental decoding. We'll get shape (B, L, V).
        with mx.stream(generation_stream):
            logits = model(y, cache=prompt_cache)
            # logits: shape (B, L, vocab_size) -> each row's last token is at index -1
            # We only need the last token's logits for sampling
            logits = logits[:, -1, :]  # shape: (B, vocab_size)

            mx.async_eval(logits)

        # We'll do a per-sequence loop for sampling and store the results
        next_tokens_list = []
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampled_tokens = sampler(logprobs).tolist()
        for i in range(batch_size):
            if ended[i]:
                # Already ended, ignore. Keep the same token as before to keep shape consistent
                # This effectively stops generating for that sequence.
                # You could choose a special "padding" token, but we just re-use y[i,0].
                next_tokens_list.append(y[i, 0])
                continue

            # 3. Sample the next token
            sampled_token_val = sampled_tokens[i]

            # 4. Check if EOS
            if sampled_token_val in tokenizer.eos_token_ids:
                ended[i] = True

            next_tokens_list.append(sampled_token_val)
            if not ended[i]:
                tokens_so_far[i].append(sampled_token_val)

        # Convert next_tokens_list -> (B, 1) array
        next_tokens = mx.array(next_tokens_list).reshape(batch_size, 1)

        # Update tokens_so_far for those that haven't ended
        #tokens_so_far = mx.concatenate([tokens_so_far, next_tokens], axis=1)


        # Prepare for the next iteration
        y = next_tokens
        n += 1

        # If all sequences are ended, break
        if all(ended):
            break

        if n % 256 == 0:
            mx.metal.clear_cache()

    # Done with generation
    generation_time = time.perf_counter() - tic
    total_generated_tokens = sum(len(seq) for seq in tokens_so_far)
    # decode all sequences
    for i in range(batch_size):
        to_decode = tokens_so_far[i]
        decoded_texts[i] = tokenizer.decode(to_decode)

            
            

    # Optionally print verbose info
    if verbose:
        for i, txt in enumerate(decoded_texts):
            print("=" * 10)
            print(f"Batch {i}: {txt}")
        print("=" * 10)
        if len(decoded_texts) == 0:
            print("No text generated for this prompt.")
        else:
            # If all sequences have the same # prompt tokens (which they do in this design),
            # we can still measure TPS
            print(
                f"Prompt tokens (per sequence): {total_prompt_len}, "
                f"Prompt TPS (across all sequences): {prompt_tps:.3f}"
            )
            # generation tokens is n for each sequence in the worst case
            # (some may have ended earlier though). We'll still show a rough TPS:
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
        Generate responses in a streaming fashion.
        Yields intermediate decoded texts at each generation step.
        Each yield returns a dict with keys "type" ("update" or "final") and "decoded_texts".
        """
        from typing import List, Optional, Union
        from transformers import PreTrainedTokenizer
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        import mlx.core as mx
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)
        prompt_ids = mx.array(prompt)
        if prompt_cache is None:
            from mlx_lm import cache as lm_cache
            prompt_cache = lm_cache.make_prompt_cache(model)
        total_prompt_len = prompt_ids.shape[0]
        processed = 0
        while processed < total_prompt_len:
            chunk_end = min(processed + prefill_step_size, total_prompt_len)
            inputs_chunk = prompt_ids[processed:chunk_end]
            _ = model(inputs_chunk[None], cache=prompt_cache)
            processed = chunk_end
        const_batch = batch_size
        y = mx.repeat(prompt_ids[-1:][None, :], repeats=const_batch, axis=0)
        for c in prompt_cache:
            c.keys = mx.repeat(c.keys, repeats=const_batch, axis=0)
            c.values = mx.repeat(c.values, repeats=const_batch, axis=0)
        tokens_so_far = [[] for _ in range(const_batch)]
        n = 0
        while n < max_tokens:
            with mx.new_stream(mx.default_device()):
                logits = model(y, cache=prompt_cache)
                logits = logits[:, -1, :]
                mx.async_eval(logits)
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)
            sampled = sampler(logits).tolist()
            sampled_tokens = []
            ended = [False] * const_batch
            for i in range(const_batch):
                token = sampled[i]
                if token in tokenizer.eos_token_ids:
                    ended[i] = True
                else:
                    tokens_so_far[i].append(token)
                sampled_tokens.append(token)
            y = mx.array(sampled_tokens).reshape(const_batch, 1)
            n += 1
            decoded_texts = [tokenizer.decode(seq) for seq in tokens_so_far]
            yield {"type": "update", "decoded_texts": decoded_texts}
            if all(ended):
                break
        yield {"type": "final", "decoded_texts": [tokenizer.decode(seq) for seq in tokens_so_far]}
