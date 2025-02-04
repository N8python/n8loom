from mlx_lm import load
from n8loom import Loom  # if you added the import to __init__.py
# Example Use
model, tokenizer = load("Llama-3.2-3B-Instruct-4bit")
prompt = "Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?"
root = Loom(model, tokenizer, prompt)

assistant_start = root.add_text_child("I will solve this problem step by step and be mindful of mistakes.")
assistant_start.ramify(n=8, temp=0.6, max_tokens=512, min_p=0.05)

answers = assistant_start.apply_at_leaves(
	lambda x: x.ramify("\n...Wait. I need to look at the problem again. Let's think about what I could've gotten wrong. I could've") if x.terminal else None,
	lambda x: x.ramify(n=2, temp=0.6, max_tokens=512, min_p=0.05),
	lambda x: x.crown()
)

for i, answer in enumerate(answers):
	print(f"Answer {i+1}:\n{answer}\n")
