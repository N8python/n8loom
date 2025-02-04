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
      - [`class Loom(Heddle)`](#class-loomheddle)
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

Below is an example (from `examples/reflection.py`) demonstrating how to load a model, create a reasoning tree (a *Loom*), and expand it with multiple potential answers:

```python
from mlx_lm import load
from n8loom import Loom  # Make sure n8loom is importable (e.g., added to __init__.py)

# Load the model and tokenizer
model, tokenizer = load("Llama-3.2-3B-Instruct-4bit")

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

#### To run the server locally:

```bash
python src/n8loom/examples/server.py
```

## API Documentation

### Core Classes (`loom.py`)

#### `class Heddle`
A *Heddle* represents a node in a reasoning tree.

- **Attributes:**
  - `text`: The text content of this node.
  - `tokens`: Tokenized representation (list of token IDs).
  - `frag`: A list of cache fragments (KVFrag) corresponding to this node.
  - `children`: List of child Heddle nodes.
  - `parent`: Reference to the parent Heddle node.
  - `terminal`: Boolean flag indicating if no further expansion is allowed.

- **Key Methods:**
  - `clip(token_limit: int)`: Clips the node’s tokens and cache fragments to a specified token limit.
  - `trim(token_trim: int)`: Removes the last N tokens from the node and resets its children.
  - `add_child(child: Heddle)`: Adds an existing node as a child.
  - `add_text_child(text: str)`: Creates and adds a new child node from a text prompt.
  - `ramify(arg: Optional[Union[str, List[str]]] = None, **kwargs)`: Expands the node by either adding text children or by sampling new responses using model generation.
  - `make_children(...)`: Generates multiple children (branches) using batched model generation.
  - `get_prefix_cache()`: Retrieves the cumulative cache from the root node up to the current node.
  - Other utility methods for traversing, displaying, and counting nodes.

#### `class Loom(Heddle)`
A *Loom* is a specialized Heddle that serves as the root of a reasoning tree. It is typically initialized with a user prompt and can optionally use a chat template.

- **Key Differences:**
  - On initialization, it processes the prompt and applies a chat template if available.
  - Its `display_text()` method shows a formatted prompt-response view.

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
