# N8Loom: For Fast Tree-of-Thought Inference

N8Loom is a Python library built on top of [mlx_lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) and [Transformers](https://huggingface.co/transformers/) that enables structured, tree-based interactions with language models. It's main selling point is its KV cache tree: it stores individual 'fragments' of the KV cache at each node in the tree, which it then concatenates to form the full cache when generating text from a node in the tree. This allows maintaining the cache of many different branches of the tree in parallel, and then merging them together when needed. This gives the inference improvements of caching without the overhead of storing the entire prefix cache at each node.

Below is a visualization of the critical difference when generating from a single node in the tree - a standard prompt cache must recompute the cache for parent nodes each time, while the KV cache tree can simply concatenate the cache fragments stored at each node to form the full cache.

| Standard Prompt Cache  | Loom Cache  |
|------------------------|------------|
| ![](media/default.gif) | ![](media/n8loom.gif) |


It additionally provides a set of utilities to manage internal model caches, generate text in parallel and stream mode, and build reasoning trees where each node represents a model “thought” (called a *Heddle*) that can branch off into multiple potential continuations. The library also includes a FastAPI server example for deploying a web service.

## Table of Contents

- [N8Loom: For Fast Tree-of-Thought Inference](#n8loom-for-fast-tree-of-thought-inference)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Basic Script Example](#basic-script-example)
    - [Running the FastAPI Server](#running-the-fastapi-server)
      - [To run the server locally:](#to-run-the-server-locally)
  - [API Documentation](#api-documentation)
    - [Core Classes (`loom.py`)](#core-classes-loompy)
      - [`class Heddle`](#class-heddle)
      - [`class Loom` *(subclass of Heddle)*](#class-loom-subclass-of-heddle)
    - [Generation Utilities (`utils.py`)](#generation-utilities-utilspy)
    - [Cache Utilities (`cache_utils.py`)](#cache-utilities-cache_utilspy)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Overview

N8Loom makes it easy to interact with language models by allowing you to:
- **Cache and manipulate intermediate model states.**  
  Utilities in `cache_utils.py` extract, clip, and fuse key-value caches (KV caches) for each model layer.
- **Create and manage reasoning trees.**  
  The core abstractions are the `Heddle` and `Loom` classes (in `loom.py`), which represent individual reasoning nodes and the overall prompt tree respectively.
- **Generate responses in batches and streams.**  
  Use the functions in `utils.py` to prefill caches, sample model outputs in parallel, or yield token-by-token updates.

## Installation

Ensure you have Python 3.7+ installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install:

```bash
pip install n8loom
```

## Usage

### Basic Script Example

Note that n8loom only works w/ the llama architecture.

Below is an example (from `examples/reflection.py`) demonstrating how to load a model, create a reasoning tree (a *Loom*), and expand it with multiple potential answers:

```python
from mlx_lm import load
from n8loom import Loom, load_for_loom

# Load the model and tokenizer
model, tokenizer = load_for_loom("Llama-3.2-3B-Instruct-4bit")

# Define a problem prompt
prompt = (
    "Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month "
    "for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. "
    "He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. "
    "If he mows 4 lawns, how many driveways did he shovel?"
)

# Create a Loom (root of the reasoning tree)
root = Loom(model, tokenizer, prompt)

# Add an initial text child to guide the model's reasoning
assistant_start = root.add_text_child("I will solve this problem step by step and be mindful of mistakes.")

# Expand the reasoning tree by generating 8 potential response branches
assistant_start.ramify(n=8, temp=0.6, max_tokens=512, min_p=0.05)

# Apply further reasoning to all leaf nodes, incorporating reflection
answers = assistant_start.apply_at_leaves(
    lambda x: x.ramify("\n...Wait. I need to look at the problem again. Let's think about what I could've gotten wrong. I could've")
    if x.terminal else None,
    lambda x: x.ramify(n=2, temp=0.6, max_tokens=512, min_p=0.05),
    lambda x: x.crown()
)

# Print the generated answers
for i, answer in enumerate(answers):
    print(f"Answer {i+1}:\n{answer}\n")
```

### Running the FastAPI Server

The library also comes with an example FastAPI server (see `examples/server.py`) that exposes endpoints to manage models, create looms, expand nodes, and export/import reasoning trees.

Make sure you have an mlx-lm model in the root directory (the parent directory of src). You can quickly do this w/:

```bash
pip install huggingface_hub hf_transfer

huggingface-cli download --local-dir Llama-3.2-3B-Instruct-4bit mlx-community/Llama-3.2-3B-Instruct-4bit
```

#### To run the server locally:

```bash
python src/n8loom/examples/server.py
```

## API Documentation

### Core Classes (`loom.py`)
#### `class Heddle`

A *Heddle* represents a node in a reasoning tree. Each node contains a segment of text, its tokenized form, cache fragments from the model, and potential child nodes. This structure enables branching reasoning and interactive exploration of model-generated responses.

- **Attributes:**
  - `model`: The language model (an instance of `nn.Module`) used for generating responses and cache fragments.
  - `tokenizer`: The tokenizer (a `PreTrainedTokenizer` or `TokenizerWrapper`) used to encode text into tokens and decode tokens back to text.
  - `text`: The text content of this node.
  - `tokens`: The tokenized representation (a list of token IDs) for the node’s text.
  - `frag`: A list of cache fragments (`KVFrag`) that store model cache information corresponding to the tokens.
  - `children`: A list of child Heddle nodes representing subsequent branches in the reasoning tree.
  - `parent`: A reference to the parent Heddle node (or `None` if this node is the root).
  - `terminal`: A Boolean flag indicating whether further expansion (generation) is disallowed.

- **Constructor:**
  - `__init__(model, tokenizer, text, frags, children, parent=None, trim_toks=1)`
    - **Purpose:** Initializes a new Heddle node.
    - **Parameters:**
      - `model`: The language model to use.
      - `tokenizer`: The tokenizer to encode/decode text.
      - `text`: The text prompt for the node.
      - `frags`: An optional list of pre-computed cache fragments. If `None`, the fragments are generated based on the text.
      - `children`: An optional list of child nodes (defaults to an empty list if not provided).
      - `parent`: The parent node (defaults to `None` for the root).
      - `trim_toks`: The number of initial tokens to trim from the token list (default is 1).

- **Key Methods:**
  - `clip(token_limit: int)`
    - **Purpose:** Clips the node’s tokens, text, and cache fragments to a specified token limit.
    - **Details:**
      - If `token_limit` is negative, it retains `len(tokens) + token_limit` tokens.
      - If the number of tokens exceeds the limit, the node’s tokens are truncated, the text is updated via decoding, the cache fragments are clipped accordingly, and all children are removed.
    - **Returns:** The current Heddle instance.
  
  - `trim(token_trim: int)`
    - **Purpose:** Removes the last `token_trim` tokens from the node.
    - **Details:** Internally calls `clip` with a negative token limit.
    - **Returns:** The current Heddle instance.
  
  - `to_leaf()`
    - **Purpose:** Converts the current node into a leaf node by removing all its children.
    - **Returns:** The current Heddle instance.
  
  - `add_child(child: Heddle)`
    - **Purpose:** Adds an existing Heddle node as a child.
    - **Details:** Also sets the added child’s `parent` attribute to this node.
    - **Returns:** The added child node.
  
  - `add_text_child(text: str)`
    - **Purpose:** Creates a new child node from a text prompt and adds it as a child.
    - **Returns:** The newly created child node.
  
  - `remove_child(child: Heddle)`
    - **Purpose:** Removes a specified child node from the current node.
    - **Returns:** The removed child node.
  
  - `get_prefix_cache() -> List[KVCache]`
    - **Purpose:** Retrieves the cumulative cache from the root node up to the current node.
    - **Details:** Collects and fuses cache fragments from all ancestor nodes to form a complete context cache.
    - **Returns:** A list of fused `KVCache` objects.
  
  - `make_children(n: int = 4, temp: float = 0.8, max_tokens: int = 8, min_p: float = 0.05, **kwargs)`
    - **Purpose:** Generates multiple child nodes using batched model generation.
    - **Details:**
      - Uses the current node’s cumulative cache as context.
      - Calls a batched generation routine to generate new text completions.
      - For each generated text, a new child is created.
      - If generation signals termination (via an `ended` flag), the child is marked as terminal.
      - Clears the model cache after generation.
    - **Parameters:**
      - `n`: Number of children to generate.
      - `temp`: Sampling temperature.
      - `max_tokens`: Maximum number of tokens to generate for each child.
      - `min_p`: Minimum probability threshold for generation.
    - **Returns:** A list of newly created child nodes.
  
  - `ramify(arg: Optional[Union[str, List[str]]] = None, **kwargs)`
    - **Purpose:** Expands the node by either adding text children or by generating new responses.
    - **Details:**
      - If `arg` is a string, creates a single child using that text.
      - If `arg` is a list of strings, creates a child for each string.
      - If `arg` is not provided, uses model generation:
        - If `stream=True` is provided in `kwargs`, streaming generation is used via `make_child_stream`.
        - Otherwise, batched generation is performed via `make_children`.
    - **Returns:** A single child, a list of children, or a streaming generator, depending on the input.
  
  - `make_child_stream(n: int = 4, temp: float = 0.8, max_tokens: int = 8, min_p: float = 0.05, **kwargs)`
    - **Purpose:** Generates child nodes using a streaming generation process.
    - **Details:**
      - Yields incremental updates (as dictionaries) from the generation process.
      - Upon receiving a final update (indicated by `"type": "final"`), creates child nodes from the generated texts.
      - Clears the model cache after finalization.
    - **Parameters:**
      - `n`: Number of children to generate.
      - `temp`: Sampling temperature.
      - `max_tokens`: Maximum number of tokens to generate for each child.
      - `min_p`: Minimum probability threshold for generation.
    - **Yields:** Updates (as dictionaries) during the generation stream.
    - **Returns:** A list of newly created child nodes after the final update.
  
  - `get_prefix_text(exclude: int = 0) -> str`
    - **Purpose:** Retrieves concatenated text from all ancestor nodes (including the current node).
    - **Parameters:**
      - `exclude`: Number of initial nodes to exclude from the prefix (default is 0).
    - **Returns:** A single string of the concatenated prefix text.
  
  - `get_display_text(exclude: int = 0) -> str`
    - **Purpose:** Similar to `get_prefix_text` but uses each node's `display_text()` method.
    - **Parameters:**
      - `exclude`: Number of initial nodes to exclude (default is 0).
    - **Returns:** A concatenated string suitable for display.
  
  - `crown() -> str`
    - **Purpose:** Returns the cumulative text from the root node up to this node, excluding the root’s text.
  
  - `display_text() -> str`
    - **Purpose:** Returns the text content of the current node.
    - **Details:** This method may be overridden in subclasses to provide formatted or additional context.
  
  - `get_prefix_tokens(exclude: int = 0) -> List[int]`
    - **Purpose:** Retrieves a concatenated list of token IDs from all ancestor nodes (including the current node).
    - **Parameters:**
      - `exclude`: Number of initial nodes to exclude (default is 0).
    - **Returns:** A list of token IDs.
  
  - `apply_all_children(func: Callable[[Heddle], Any], apply_self: bool = False, leaves_only: bool = False) -> List[Any]`
    - **Purpose:** Applies a given function to all descendant nodes.
    - **Parameters:**
      - `func`: A function that takes a Heddle node as input.
      - `apply_self`: Whether to apply the function to the current node as well.
      - `leaves_only`: If True, applies the function only to leaf nodes.
    - **Returns:** A list of results from applying the function to the nodes.
  
  - `at_all_leaves(func: Callable[[Heddle], Any]) -> List[Any]`
    - **Purpose:** Convenience method to apply a function only to all leaf nodes.
    - **Returns:** A list of results from applying the function to each leaf.
  
  - `apply_at_leaves(*funcs: Callable[[Heddle], Any])`
    - **Purpose:** Sequentially applies multiple functions to all leaf nodes.
    - **Details:** All functions except the last are applied for their side effects; the final function’s results are returned.
    - **Returns:** A list of results from the final function applied to each leaf.
  
  - `get_all_children(depth: int = 0) -> List[Heddle]`
    - **Purpose:** Recursively retrieves all descendant nodes in the subtree.
    - **Parameters:**
      - `depth`: Used internally to decide whether to include the current node (included if `depth > 0`).
    - **Returns:** A flat list of all descendant Heddle nodes.
  
  - `get_all_leaves() -> List[Heddle]`
    - **Purpose:** Retrieves all leaf nodes (nodes with no children) in the subtree.
    - **Returns:** A list of leaf nodes.
  
  - `count_children()`
    - **Purpose:** Counts the total number of nodes in the subtree rooted at this node (including itself).
    - **Returns:** An integer count of the nodes.
  
  - `count_leaves()`
    - **Purpose:** Counts the total number of leaf nodes in the subtree.
    - **Details:** If there are no children, returns 1 (the current node itself).
    - **Returns:** An integer count of the leaf nodes.
  
  - `__repr__()`
    - **Purpose:** Returns a string representation of the Heddle node.
    - **Returns:** A string displaying the node’s text and a summary of its children.
#### `class Loom` *(subclass of Heddle)*

*Loom* is a specialized subclass of Heddle used as the root node in chat-based or conversational settings.

- **Additional Attributes:**
  - `user_prompt`: The original prompt provided by the user.
  - `chat_template_used`: A flag indicating whether a chat template was applied to the prompt.
  
- **Constructor:**
  - `__init__(model, tokenizer, prompt)`
    - **Purpose:** Initializes a Loom instance.
    - **Details:**
      - Stores the original user prompt.
      - Attempts to apply a chat template via `tokenizer.apply_chat_template` (if available).
      - Calls the Heddle constructor with appropriate parameters.
  
- **Overridden Methods:**
  - `display_text() -> str`
    - **Purpose:** Returns formatted text for display.
    - **Details:** If a chat template was used, it prefixes the output with "Prompt:" followed by the original user prompt and then a "Response:" section. Otherwise, it returns the plain text as defined in Heddle.

### Generation Utilities (`utils.py`)

- **`prompt_to_cache(model, tokenizer, prompt_ids, c=None, prefill_step_size=512)`**  
  Processes a prompt through the model in steps, filling the key-value cache.

- **`generate_batched(...)`**  
  Generates multiple responses in parallel from a prompt. Returns generated texts, updated caches, token counts, and flags indicating termination.

- **`generate_batched_stream(...)`**  
  Similar to `generate_batched`, but yields incremental generation updates in a streaming manner.


### Cache Utilities (`cache_utils.py`)

- **`frag_cache(cache: List[KVCache], start_idx: int = 0, end_idx: int = -1) -> List[KVFrag]`**  
  Extracts key-value cache fragments from each layer between the specified indices.

- **`clip_frag(frags: List[KVFrag], token_limit: int) -> List[KVFrag]`**  
  Clips the keys and values in cache fragments to the given token limit.

- **`frag_batch_gen(cache: List[KVCache], total_prompt_len: int, generated_lengths: List[int]) -> List[List[KVFrag]]`**  
  Creates cache fragments for each batch instance based on prompt and generated token lengths.

- **`fuse_cache_frags(frags: List[List[KVFrag]]) -> List[KVCache]`**  
  Merges a list of cache fragments into full KVCache objects, concatenating along the sequence dimension.

## Contributing

Contributions to enhance N8Loom (e.g., new features, bug fixes, or improved documentation) are very welcome. Please file issues or submit pull requests on the project's repository.

## License

This project is licensed under the CC0 License.

## Acknowledgements

- **mlx_lm:** The library builds upon the efficient language model framework provided by mlx_lm.
- **Transformers:** For model and tokenizer support.
- **FastAPI & Uvicorn:** For providing a lightweight web server example.

---

This documentation and the included examples should help you get started with building interactive, tree-based language model applications using N8Loom. Happy coding!
