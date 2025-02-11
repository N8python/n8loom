import re
import math
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from mlx_lm import load
from n8loom import Loom, load_for_loom 
import mlx.core as mx

# -------------------------------------------------------------------
# Load the model and tokenizer once (this may take some time)
# -------------------------------------------------------------------
print("Loading model ...")
model, tokenizer = load_for_loom("Llama-3.2-3B-Instruct-4bit")
print("Model loaded.\n")

# -------------------------------------------------------------------
# Function to run a single trial with a given answer_count.
# It returns the execution time and the peak memory usage (in GB).
# -------------------------------------------------------------------
def run_trial(answer_count):
    start_time = time.perf_counter()

    prompt = (
        "Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month "
        "for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. "
        "He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. "
        "If he mows 4 lawns, how many driveways did he shovel?"
    )

    # Build the chain using Loom
    root = Loom(model, tokenizer, prompt)
    assistant_start = root.add_text_child("I will solve this problem step by step and be mindful of mistakes.")
    assistant_start.ramify(n=answer_count, temp=0.6, max_tokens=512, min_p=0.05)

    answers = assistant_start.apply_at_leaves(
        lambda x: x.ramify("\n...Alright, I'll put my final answer between <answer> XML tags. My answer is <answer>") if x.terminal else None,
        lambda x: x.ramify(n=1, temp=0.0, max_tokens=32, min_p=0.05),
        lambda x: x.crown()
    )

    # Use a regex to capture everything between <answer> and </answer>
    pattern_content = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    parsed_final_answers = []
    for answer in answers:
        match = pattern_content.search(answer)
        if match:
            content = match.group(1)
            number_match = re.search(r'\d+', content)
            if number_match:
                try:
                    number = int(number_match.group(0))
                    parsed_final_answers.append(number)
                except ValueError:
                    parsed_final_answers.append(float('nan'))
            else:
                parsed_final_answers.append(float('nan'))
        else:
            parsed_final_answers.append(float('nan'))

    # (Optional) Aggregate answers for majority voting.
    answer_counts = {}
    for answer in parsed_final_answers:
        if math.isnan(answer):
            continue
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)

    # Get peak memory usage (in GB)
    peak_memory = mx.metal.get_peak_memory() / 1e9

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    return execution_time, sorted_answers, peak_memory

# -------------------------------------------------------------------
# Benchmark function that runs trials for answer_count values in powers
# of 2 from 1 to 128. For each answer_count, it repeats trials until either
# the standard error of execution time falls below 2 seconds or 10 trials are done.
#
# After each trial, the script waits 10 seconds to help avoid thermal throttling.
#
# Finally, the function saves the benchmark results to a CSV file and
# creates a dual-axis plot showing both mean response time and mean peak memory.
# -------------------------------------------------------------------
def benchmark():
    # Answer counts: 1, 2, 4, ..., 128
    answer_counts_list = [2 ** i for i in range(0, 7)]
    results = []  # Each entry: (answer_count, mean_time, time_std_err, trials, mean_peak_memory, memory_std_err)
    
    for count in answer_counts_list:
        trial_times = []
        trial_memories = []
        trial = 0
        print(f"Starting trials for answer_count = {count}")
        while True:
            trial += 1
            print(f"  Running trial {trial}...")
            exec_time, sorted_answers, peak_memory = run_trial(count)
            trial_times.append(exec_time)
            trial_memories.append(peak_memory)
            print(f"    Execution time: {exec_time:.2f} seconds, Peak memory: {peak_memory:.2f} GB")
            
            # Wait 60 seconds after each trial to avoid thermal throttling
            time.sleep(20 * (trial + 1))
            
            # Stop if 10 trials have been run.
            if trial >= 10:
                break
            # If more than one trial has been done, check if the standard error is below 2 seconds.
            if len(trial_times) > 1:
                std_err = np.std(trial_times, ddof=1) / np.sqrt(len(trial_times))
                if std_err < 2.0:
                    break
        
        mean_time = np.mean(trial_times)
        time_std_err = np.std(trial_times, ddof=1) / np.sqrt(len(trial_times)) if len(trial_times) > 1 else 0.0
        mean_memory = np.mean(trial_memories)
        memory_std_err = np.std(trial_memories, ddof=1) / np.sqrt(len(trial_memories)) if len(trial_memories) > 1 else 0.0
        
        results.append((count, mean_time, time_std_err, trial, mean_memory, memory_std_err))
        print(f"Results for answer_count = {count}: Mean Time = {mean_time:.2f}s, Std Error = {time_std_err:.2f}s, Trials = {trial}, Mean Peak Memory = {mean_memory:.2f} GB\n")
    
    # Save results to a CSV file.
    csv_filename = "benchmark_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["answer_count", "mean_time", "std_error", "trials", "mean_peak_memory", "memory_std_error"])
        for row in results:
            writer.writerow(row)
    print(f"Saved benchmark results to {csv_filename}")
    
    # Prepare data for plotting.
    counts = [r[0] for r in results]
    mean_times = [r[1] for r in results]
    time_errors = [r[2] for r in results]
    mean_memories = [r[4] for r in results]
    memory_errors = [r[5] for r in results]
    
    # Create a dual-axis plot.
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot mean response time (left y-axis)
    ax1.errorbar(counts, mean_times, yerr=time_errors, fmt='o-', capsize=5, color='tab:blue', label='Mean Time (s)')
    ax1.set_xlabel("Answer Count")
    ax1.set_ylabel("Mean Response Time (seconds)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log', base=2)
    
    # Create a second y-axis for peak memory.
    ax2 = ax1.twinx()
    ax2.errorbar(counts, mean_memories, yerr=memory_errors, fmt='s--', capsize=5, color='tab:red', label='Mean Peak Memory (GB)')
    ax2.set_ylabel("Mean Peak Memory (GB)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Title and grid.
    fig.suptitle("Benchmark: Answer Count vs Mean Response Time and Peak Memory")
    ax1.grid(True, which="both", ls="--")
    
    # Add a legend combining both axes.
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", bbox_to_anchor=(0.1, 0.9))
    
    # Save the plot.
    plot_filename = "benchmark_plot.png"
    plt.savefig(plot_filename)
    print(f"Saved benchmark plot to {plot_filename}")
    plt.show()

# -------------------------------------------------------------------
# Run the benchmark if this script is executed as the main module.
# -------------------------------------------------------------------
if __name__ == "__main__":
    benchmark()
