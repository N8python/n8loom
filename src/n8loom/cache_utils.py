
from collections import namedtuple
from mlx_lm.models.cache import KVCache
import mlx.core as mx
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
KVFrag = namedtuple("KVFrag", ["keys", "values"])
def mx_copy(x: mx.array) -> mx.array:
	return mx.array(np.array(x))
def frag_cache(cache: List[KVCache], start_idx: int = 0, end_idx: int = -1) -> List[KVFrag]:
	"""Extracts and converts a slice of key-value pairs from model layer caches into fragments.

	Args:
		cache: List of KVCache objects, one per model layer, containing cached key-value pairs
		start_idx: Starting index for extraction (default: 0)
		end_idx: Ending index for extraction (default: -1)

	Returns:
		List of KVFrag objects, one per layer, each containing the extracted keys and values
		arrays from the specified index range

	Example:
		>>> layer_caches = [KVCache(...), KVCache(...)]  # List of caches for each layer
		>>> fragments = frag_cache(layer_caches, 0, 10)  # Get first 10 positions
	"""
	frags = []
	for layer_cache in cache:
		keys = mx_copy(layer_cache.keys[:, :, start_idx:end_idx])
		values = mx_copy(layer_cache.values[:, :, start_idx:end_idx])
		frags.append(KVFrag(keys, values))
	return frags
def clip_frag(frags: List[KVFrag], token_limit: int) -> List[KVFrag]:
	"""Clips a list of key-value fragments to a specified token limit.

	Args:
		frags: List of KVFrag objects - one per layer - to clip
		token_limit: Maximum number of tokens to retain in each fragment

	Returns:
		List of KVFrag objects, each containing the clipped keys and values arrays
	"""
	clipped_frags = []
	for frag in frags:
		keys = mx_copy(frag.keys[:, :, :token_limit])
		values = mx_copy(frag.values[:, :, :token_limit])
		clipped_frags.append(KVFrag(keys, values))
	return clipped_frags
def frag_batch_gen(cache: List[KVCache], total_prompt_len: int, generated_lengths: List[int]) -> List[List[KVFrag]]:
	frags = []
	B = cache[0].keys.shape[0]
	for i in range(B):
		batch_frags = []
		for layer_cache in cache:
			keys = mx_copy(layer_cache.keys[i:i+1, :, total_prompt_len:total_prompt_len + generated_lengths[i]])
			values = mx_copy(layer_cache.values[i:i+1, :, total_prompt_len:total_prompt_len + generated_lengths[i]])
			batch_frags.append(KVFrag(keys, values))
		frags.append(batch_frags)
	return frags
	
def fuse_cache_frags(frags: List[List[KVFrag]], offset: int) -> List[KVCache]:
	"""Fuses a list of key-value fragments into a list of model layer caches.

	Args:
		frags: List of lists of KVFrag objects - first dimension is the layer index, second is the list of fragments to merge

	Returns:
		List of KVCache objects, one per model layer, containing the fused key-value pairs from the fragments, concatenated along the sequence dimension.
	
	Example:
		>>> fragments = [[KVFrag(...), KVFrag(...)], [KVFrag(...), KVFrag(...)]]
		>>> layer_caches = fuse_cache_frags(fragments)
	"""
	caches = []
	for layer_frags in frags:
		keys = mx.concat([
			frag.keys 
			for i, frag in enumerate(layer_frags)
		], axis=2)

		values = mx.concat([
			frag.values 
			for i, frag in enumerate(layer_frags)
		], axis=2)		
		cache = KVCache()
		cache.keys = keys
		cache.values = values
		cache.offset = keys.shape[2]
		caches.append(cache)
	return caches
	