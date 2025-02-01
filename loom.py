from __future__ import annotations  # This allows using Heddle directly in annotations
from typing import Any, Dict, List, Optional, Union, Callable, Set
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.models.cache import KVCache
from mlx_lm import load, generate
from utils import generate_batched, generate_batched_stream, prompt_to_cache
import numpy as np
import copy
from collections import namedtuple
from cache_utils import KVFrag, frag_cache, frag_batch_gen, fuse_cache_frags, clip_frag

class Heddle:
	text: str
	frag: List[KVFrag]
	tokens: List[int]
	children: List[Heddle]
	parent: Optional[Heddle]
	terminal: bool
	def __init__(self, model: nn.Module, tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper], text: str, frags: Optional[List[KVFrag]], children: Optional[List[Heddle]], parent: Optional[Heddle] = None):
		self.model = model
		self.tokenizer = tokenizer
		self.text = text
		self.parent = parent
		self.tokens = tokenizer.encode(text)[1:]
		if frags is None:
			c = None
			if self.parent is None:
				c, l = prompt_to_cache(model, tokenizer, self.tokens)
				c = frag_cache(c, 0, l)
			else:
				parent_cache = self.parent.get_prefix_cache()
				p_len = parent_cache[0].offset
				c, l = prompt_to_cache(model, tokenizer, self.tokens, c=parent_cache)
				c = frag_cache(c, p_len, p_len + l)
			frags = c
		self.frag = frags
		if children is None:
			children = []
		self.children = children
		for child in self.children:
			child.parent = self
		self._cache = None
		self.terminal = False
	def clip(self, token_limit: int):
		if token_limit < 0:
			token_limit = max(len(self.tokens) + token_limit, 0)
		if len(self.tokens) > token_limit:
			self.tokens = self.tokens[:token_limit]
			self.text = self.tokenizer.decode(self.tokens)
			self.frag = clip_frag(self.frag, token_limit)
			self.children = []
		return self
	def trim(self, token_trim: int):
		return self.clip(-token_trim)
	def to_leaf(self):
		if self.children:
			self.children = []
		return self
	def add_child(self, child: Heddle):
		self.children.append(child)
		child.parent = self
		return child
	def add_text_child(self, text: str):
		child = Heddle(self.model, self.tokenizer, text, None, [], self)
		self.add_child(child)
		return child
	def remove_child(self, child: Heddle):
		self.children.remove(child)
		child.parent = None
		return child
	def get_prefix_cache(self) -> List[KVCache]:
		parents = [self]
		parent = self.parent
		while parent is not None:
			parents.append(parent)
			parent = parent.parent
		parents.reverse()
		cache = [[] for _ in range(len(self.frag))]
		for parent in parents:
			for i, frag in enumerate(parent.frag):
				cache[i].append(frag)
		fused = fuse_cache_frags(cache)
		mx.eval(fused)
		return fused
	def make_children(self, n: int = 4, temp: float = 0.8, max_tokens: int = 8):
		if self.terminal:
			return
		c = self.get_prefix_cache()
		decoded_texts, prompt_cache, total_prompt_len, generated_lengths, ended = generate_batched(self.model, self.tokenizer, prompt=self.tokens, batch_size=n, prompt_cache=c, verbose=False, temp=temp, max_tokens=max_tokens)
		fragments = frag_batch_gen(prompt_cache, total_prompt_len, generated_lengths)
		made_kids = []
		for i in range(len(fragments)):
			child = Heddle(self.model, self.tokenizer, decoded_texts[i], fragments[i], [])
			if ended[i]:
				child.terminal = True
			self.add_child(child)
			made_kids.append(child)
		return made_kids
	def ramify(self, arg=None, **kwargs):
		if isinstance(arg, str):
			return self.add_text_child(arg)
		elif isinstance(arg, list) and all([isinstance(x, str) for x in arg]):
			children = [self.add_text_child(text) for text in arg]
			return children
		else:
			if kwargs.get('stream', False):
				from utils import generate_batched_stream
				c = self.get_prefix_cache()
				return generate_batched_stream(
					self.model,
					self.tokenizer,
					self.tokens,
					batch_size=kwargs.get('n', 4),
					prompt_cache=c,
					verbose=False,
					temp=kwargs.get('temp', 0.8),
					max_tokens=kwargs.get('max_tokens', 8)
				)
			else:
				return self.make_children(**kwargs)
	def get_prefix_text(self, exclude: int = 0) -> str:
		parents = [self]
		parent = self.parent
		while parent is not None:
			parents.append(parent)
			parent = parent.parent
		parents.reverse()
		return "".join([parent.text for parent in parents[exclude:]])
	def get_display_text(self, exclude: int = 0) -> str:
		parents = [self]
		parent = self.parent
		while parent is not None:
			parents.append(parent)
			parent = parent.parent
		parents.reverse()
		return "".join([parent.display_text() for parent in parents[exclude:]])
	def crown(self) -> str:
		return self.get_prefix_text(1)
	def display_text(self) -> str:
		return self.text
	def get_prefix_tokens(self, exclude: int = 0) -> List[int]:
		parents = [self]
		parent = self.parent
		while parent is not None:
			parents.append(parent)
			parent = parent.parent
		parents.reverse()
		return [token for parent in parents[exclude:] for token in parent.tokens]
	def apply_all_children(self, func: Callable[[Heddle], Any], apply_self=False, leaves_only=False) -> List[Any]:
		"""
		Applies a function to all children nodes in the tree up to a maximum depth using iteration.
		
		Args:
			func: A function that takes a Heddle object as input and returns any value
			apply_self: Whether to apply the function to the root node
			max_depth: Maximum depth to traverse (None for unlimited)
			
		Returns:
			List of results from applying the function to each child
		"""
		results = []
		if apply_self:
			results.append(func(self))
		pre_children = self.get_all_children()
		for child in pre_children:
			if leaves_only and child.children:
				continue
			results.append(func(child))
		return results
	def at_all_leaves(self, func: Callable[[Heddle], Any]) -> List[Any]:
		return self.apply_all_children(func, apply_self=False, leaves_only=True)
	def apply_at_leaves(self, *funcs: Callable[[Heddle], Any]):
		for func in funcs[:-1]:
			self.at_all_leaves(func)
		return self.at_all_leaves(funcs[-1])
	def get_all_children(self, depth=0) -> List[Heddle]:
		s = []
		if depth > 0:
			s.append(self)
		for child in self.children:
			s.extend(child.get_all_children(depth + 1))
		return s
	def get_all_leaves(self) -> List[Heddle]:
		return [child for child in self.get_all_children() if not child.children]
	def count_children(self):
		return 1 + sum([child.count_children() for child in self.children])
	def count_leaves(self):
		return sum([child.count_leaves() for child in self.children]) or 1
	def __repr__(self):
		return f"Heddle({self.text}, {self.children})"

class Loom(Heddle):
	def __init__(self, model, tokenizer, prompt):
		self.user_prompt = prompt
		messages = [{"role": "user", "content": prompt}]
		prompt = tokenizer.apply_chat_template(
			messages, add_generation_prompt=True, tokenize=False
		)
		super().__init__(model, tokenizer, prompt, None, [])
	def display_text(self):
		return "Prompt: " + self.user_prompt + "\n\nResponse:\n\n"

"""
# Example Use
model, tokenizer = load("Llama-3.2-3B-Instruct-4bit")
prompt = "Write a 4 chan greentext"
root = Loom(model, tokenizer, prompt)

assistant_start = root.add_text_child("> be me, ai language model\n")
assistant_start.ramify(n=2, temp=0.8, max_tokens=32)

greentexts = assistant_start.apply_at_leaves(
	lambda x: x.ramify("\n> harambe returns from heaven"),
	lambda x: x.ramify(n=2, temp=0.8, max_tokens=32),
	lambda x: x.trim(10).ramify(["\n> seven angels descend from a pinhead", "\n> P = NP is solved"]),
	lambda x: x.ramify(n=2, temp=0.8, max_tokens=64),
	lambda x: x.crown()
)

for i, greentext in enumerate(greentexts):
	print(f"Greentext {i+1}:\n{greentext}\n")
"""
