from __future__ import annotations  # This allows using Heddle directly in annotations
from typing import Any, Dict, List, Optional, Union, Callable
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.models.cache import KVCache
from mlx_lm import load, generate
from .utils import generate_batched, generate_batched_stream, prompt_to_cache
import numpy as np
import copy
from collections import namedtuple
from .cache_utils import KVFrag, frag_cache, frag_batch_gen, fuse_cache_frags, clip_frag


class Heddle:
	"""
	Rewritten to use tokens as the internal representation while maintaining
	the same user-facing (text-based) API.
	"""
	model: nn.Module
	tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper]
	parent: Optional[Heddle]
	children: List[Heddle]
	frag: List[KVFrag]
	terminal: bool

	def __init__(
		self,
		model: nn.Module,
		tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
		text: str,
		frags: Optional[List[KVFrag]],
		children: Optional[List[Heddle]],
		parent: Optional[Heddle] = None,
		trim_toks: int = 1
	):
		self.model = model
		self.tokenizer = tokenizer
		self.parent = parent
		# Underlying representation is tokens only:
		if type(text) == list:
			self.tokens = text
		else:
			self.tokens = tokenizer.encode(text)[trim_toks:]
		# Store trim_toks for potential future resets:
		self._trim_toks = trim_toks

		# If no fragment was passed in, generate one:
		if frags is None:
			c = None
			if self.parent is None:
				c, l = prompt_to_cache(model, tokenizer, self.tokens, offset=0)
				c = frag_cache(c, 0, l - 1)
			else:
				parent_cache = self.parent.get_prefix_cache()
				p_len = parent_cache[0].offset
				c, l = prompt_to_cache(
					model,
					tokenizer,
					self.tokens,
					c=parent_cache,
					offset=0
				)
				c = frag_cache(c, p_len, p_len + l - 1)
			frags = c

		self.frag = frags

		# Children
		if children is None:
			children = []
		self.children = children
		for child in self.children:
			child.parent = self

		self.terminal = False

	@property
	def text(self) -> str:
		"""
		Returns the decoded text from the underlying tokens.
		"""
		return self.tokenizer.decode(self.tokens)

	@text.setter
	def text(self, new_text: str):
		"""
		Sets the underlying tokens based on new_text, respecting the original trim.
		"""
		self.tokens = self.tokenizer.encode(new_text)[self._trim_toks:]

	def clip(self, token_limit: int) -> Heddle:
		if token_limit < 0:
			token_limit = max(len(self.tokens) + token_limit, 0)
		if len(self.tokens) > token_limit:
			self.tokens = self.tokens[:token_limit]
			self.frag = clip_frag(self.frag, token_limit)
			self.children = []
		return self

	def trim(self, token_trim: int) -> Heddle:
		return self.clip(-token_trim)

	def to_leaf(self) -> Heddle:
		if self.children:
			self.children = []
		return self

	def add_child(self, child: Heddle) -> Heddle:
		self.children.append(child)
		child.parent = self
		return child

	def add_text_child(self, text: str) -> Heddle:
		child = Heddle(self.model, self.tokenizer, text, None, [], self)
		self.add_child(child)
		return child

	def remove_child(self, child: Heddle) -> Heddle:
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
		fused = fuse_cache_frags(cache, offset=1)
		return fused

	def make_children(
		self,
		n: int = 4,
		temp: float = 0.8,
		max_tokens: int = 8,
		min_p: float = 0.05,
		stop_strings: List[str] = [],
		**kwargs
	) -> Optional[List[Heddle]]:
		if self.terminal:
			return None

		c = self.get_prefix_cache()
		decoded_texts, prompt_cache, total_prompt_len, generated_lengths, ended = generate_batched(
			self.model,
			self.tokenizer,
			prompt=self.tokens,
			batch_size=n,
			min_p=min_p,
			prompt_cache=c,
			verbose=False,
			temp=temp,
			max_tokens=max_tokens,
			stop_strings=stop_strings
		)
		fragments = frag_batch_gen(prompt_cache, 0, generated_lengths)
		made_kids = []
		for i in range(len(fragments)):
			child = Heddle(
				self.model,
				self.tokenizer,
				decoded_texts[i],
				fragments[i],
				[]
			)
			if ended[i]:
				child.terminal = True
			self.add_child(child)

			made_kids.append(child)
			mx.metal.clear_cache()
		return made_kids

	def ramify(self, arg=None, **kwargs):
		"""
		Convenience method to expand children. Takes either:
		  - a single string -> returns a single child
		  - a list of strings -> returns multiple children
		  - else uses model generation (batch or stream)
		"""
		if isinstance(arg, str):
			return self.add_text_child(arg)
		elif isinstance(arg, list) and all(isinstance(x, str) for x in arg):
			children = [self.add_text_child(text) for text in arg]
			return children
		else:
			if kwargs.get('stream', False):
				return self.make_child_stream(**kwargs)
			else:
				return self.make_children(**kwargs)

	def make_child_stream(
		self,
		n: int = 4,
		temp: float = 0.8,
		max_tokens: int = 8,
		min_p: float = 0.05,
		stop_strings: List[str] = [],
		**kwargs
	):
		c = self.get_prefix_cache()
		stream = generate_batched_stream(
			self.model,
			self.tokenizer,
			self.tokens,
			batch_size=n,
			prompt_cache=c,
			verbose=False,
			temp=temp,
			min_p=min_p,
			max_tokens=max_tokens,
			stop_strings=stop_strings
		)
		made_kids = []
		for update in stream:
			if update.get("type") == "final":
				final_texts = update.get("decoded_texts", [])
				generated_lengths = update.get("generated_lengths", [])
				total_prompt_len = update.get("total_prompt_len", 0)
				prompt_cache = update.get("prompt_cache", [])
				ended = update.get("ended", [])
				# free to avoid large intermediate state
				update["prompt_cache"] = None

				fragments = frag_batch_gen(prompt_cache, 0, generated_lengths)
				made_kids = []
				for i, text_str in enumerate(final_texts):
					child = Heddle(
						self.model,
						self.tokenizer,
						text_str,
						fragments[i],
						[]
					)
					if ended[i]:
						child.terminal = True
					self.add_child(child)
					made_kids.append(child)
				update["children"] = made_kids

			yield update
		return made_kids

	def get_prefix_text(self, exclude: int = 0) -> str:
		parents = [self]
		parent = self.parent
		while parent is not None:
			parents.append(parent)
			parent = parent.parent
		parents.reverse()
		return "".join([p.text for p in parents[exclude:]])

	def get_display_text(self, exclude: int = 0) -> str:
		parents = [self]
		parent = self.parent
		while parent is not None:
			parents.append(parent)
			parent = parent.parent
		parents.reverse()
		return "".join([p.display_text() for p in parents[exclude:]])

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
		return [token for p in parents[exclude:] for token in p.tokens]

	def apply_all_children(
		self,
		func: Callable[[Heddle], Any],
		apply_self: bool = False,
		leaves_only: bool = False
	) -> List[Any]:
		"""
		Applies a function to all children (and optionally self) in the subtree.

		Args:
			func: A function that takes a Heddle and returns any value.
			apply_self: Whether to apply the function to the root node itself.
			leaves_only: If True, only apply to leaf nodes (i.e., nodes with no children).

		Returns:
			A list of results from applying the function to each matching node.
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

	def get_all_children(self, depth: int = 0) -> List[Heddle]:
		"""
		Returns all descendants (the entire subtree).
		`depth` is just for internal recursion tracking.
		"""
		nodes = []
		if depth > 0:
			nodes.append(self)
		for child in self.children:
			nodes.extend(child.get_all_children(depth + 1))
		return nodes

	def get_all_leaves(self) -> List[Heddle]:
		return [child for child in self.get_all_children() if not child.children]

	def count_children(self) -> int:
		return 1 + sum([child.count_children() for child in self.children])

	def count_leaves(self) -> int:
		if not self.children:
			return 1
		return sum([child.count_leaves() for child in self.children])

	def __repr__(self):
		return f"Heddle({self.text}, {self.children})"


class Loom(Heddle):
	def __init__(self, model, tokenizer, prompt):
		self.user_prompt = prompt
		self.chat_template_used = False

		messages = [{"role": "user", "content": prompt}]
		try:
			# Attempt to apply chat template, if available
			prompt = tokenizer.apply_chat_template(
				messages,
				add_generation_prompt=True,
				tokenize=False
			)
			self.chat_template_used = True
		except:
			pass

		# Initialize base Heddle without trimming
		super().__init__(model, tokenizer, prompt, None, [], None, trim_toks=0)

	def display_text(self):
		if self.chat_template_used:
			return "Prompt: " + self.user_prompt + "\n\nResponse:\n\n"
		return super().display_text()
