import time
import matplotlib.pyplot as plt

from mlx_lm import load
import mlx.core as mx
from n8loom import Loom, load_for_loom

# Load the model and tokenizer (adjust model name if needed)
model, tokenizer = load_for_loom("Llama-3.2-3B-Instruct-4bit")

# Define the prompt (using your long story)
prompt = """
The Epic of Gilgamesh (/ˈɡɪlɡəmɛʃ/)[2] is an epic from ancient Mesopotamia. The literary history of Gilgamesh begins with five Sumerian poems about Gilgamesh (formerly read as Sumerian "Bilgames"[3]), king of Uruk, some of which may date back to the Third Dynasty of Ur (c. 2100 BCE).[1] These independent stories were later used as source material for a combined epic in Akkadian. The first surviving version of this combined epic, known as the "Old Babylonian" version, dates back to the 18th century BCE and is titled after its incipit, Shūtur eli sharrī ("Surpassing All Other Kings"). Only a few tablets of it have survived. The later Standard Babylonian version compiled by Sîn-lēqi-unninni dates to somewhere between the 13th to the 10th centuries BCE and bears the incipit Sha naqba īmuru[note 1] ("He who Saw the Deep(s)", lit. '"He who Sees the Unknown"'). Approximately two-thirds of this longer, twelve-tablet version have been recovered. Some of the best copies were discovered in the library ruins of the 7th-century BCE Assyrian king Ashurbanipal.

The first half of the story discusses Gilgamesh (who was king of Uruk) and Enkidu, a wild man created by the gods to stop Gilgamesh from oppressing the people of Uruk. After Enkidu becomes civilized through sexual initiation with Shamhat, he travels to Uruk, where he challenges Gilgamesh to a test of strength. Gilgamesh wins the contest; nonetheless, the two become friends. Together, they make a six-day journey to the legendary Cedar Forest, where they ultimately slay its Guardian, Humbaba, and cut down the sacred Cedar.[5] The goddess Ishtar sends the Bull of Heaven to punish Gilgamesh for spurning her advances. Gilgamesh and Enkidu kill the Bull of Heaven, insulting Ishtar in the process, after which the gods decide to sentence Enkidu to death and kill him by giving him a fatal illness.

In the second half of the epic, distress over Enkidu's death causes Gilgamesh to undertake a long and perilous journey to discover the secret of eternal life. Finally, he meets Utnapishtim, who with his wife were the only humans to survive the Flood triggered by the gods (cf. Athra-Hasis). Gilgamesh learns from him that "Life, which you look for, you will never find. For when the gods created man, they let death be his share, and life withheld in their own hands".[6][7]

The epic is regarded as a foundational work in religion and the tradition of heroic sagas, with Gilgamesh forming the prototype for later heroes like Heracles (Hercules) and the epic itself serving as an influence for Homeric epics.[8] It has been translated into many languages and is featured in several works of popular fiction.

Analyze the above summary of Gilgamesh and comment on what it shows about humanity. Do so in at least three paragraphs.
"""

print("Prompt length:", len(tokenizer.encode(prompt)))

# Define the batch sizes (n values) to test
n_values = [1, 2, 4, 8, 16, 32, 64, 128]
tokens_per_sec_list = []
runtimes = []
memory = []
# For each n, we generate a batch of children using a fixed max_tokens per child.
# (We assume that the generated text in each child is roughly the new tokens produced.)
for n in n_values:
    # (Optional) Reset any peak memory counters, if desired
    mx.metal.reset_peak_memory()
    # Create a new Loom instance for the current iteration
    root = Loom(model, tokenizer, prompt)
    
    # Use a fixed number of tokens to generate per child.
    max_tokens = 128
    
    start_time = time.time()
    children = root.ramify(n=n, temp=0.6, max_tokens=max_tokens, min_p=0.05)
    elapsed_time = time.time() - start_time
    
    # Count the total number of tokens generated by all children.
    # Note: Each child is initialized with its generated text.
    total_generated_tokens = sum(len(child.tokens) for child in children)
    
    # Compute tokens per second (protect against division by zero)
    tokens_sec = total_generated_tokens / elapsed_time if elapsed_time > 0 else 0
    mem_usage_gb = mx.metal.get_peak_memory() / 1e9  # convert bytes to GB
    print(f"n={n}, Total Generated Tokens: {total_generated_tokens}, "
          f"Tokens/sec: {tokens_sec:.2f}, Runtime: {elapsed_time:.2f} seconds, Peak Memory: {mem_usage_gb:.2f} GB")
    
    tokens_per_sec_list.append(tokens_sec)
    runtimes.append(elapsed_time)
    memory.append(mem_usage_gb)
    
    # Clear caches and delete the root to free memory for the next iteration.
    mx.metal.clear_cache()
    del root

# Create figure with proper subplot layout
plt.figure(figsize=(15, 5))

# Plot Tokens per Second vs n
plt.subplot(1, 3, 1)
plt.plot(n_values, tokens_per_sec_list, marker='o')
plt.xlabel('n')
plt.ylabel('Tokens per Second')
plt.title('Generation Throughput\n(Tokens/sec) vs n')

# Plot Runtime vs n
plt.subplot(1, 3, 2)
plt.plot(n_values, runtimes, marker='o', color='red')
plt.xlabel('n')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs n')

# Plot Peak Memory Usage vs n
plt.subplot(1, 3, 3)
plt.plot(n_values, memory, marker='o', color='green')
plt.xlabel('n')
plt.ylabel('Peak Memory Usage (GB)')
plt.title('Peak Memory Usage vs n')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()