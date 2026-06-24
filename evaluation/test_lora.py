from unsloth import FastLanguageModel
import torch

# 1. Load the base model and your custom adapters
print("Loading Model and LoRA Adapters...")
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen-0.5b-thinking-lora", # Pointing directly to your saved folder
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 2. Enable native inference speedups (Unsloth magic)
FastLanguageModel.for_inference(model)

# 3. Format the test question
question = "A baker has 3 bags of flour. Each bag weighs 5 kg. If he uses 4 kg to bake bread, how much flour is left in total?"

messages = [
    {"role": "system", "content": "You are a logical math solver. Think step-by-step inside <think> tags, then output the final numerical answer inside \\boxed{}."},
    {"role": "user", "content": question}
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# 4. Generate the response (Standard Greedy)
print("\n--- Generating Response ---")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    use_cache=True,
    temperature=0.1 # Greedy decoding to test pure instinct
)

decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Print just the assistant's response part
response = decoded_output.split("system\nYou are a logical math solver. Think step-by-step inside <think> tags, then output the final numerical answer inside \\boxed{}.\nuser\n" + question + "\nassistant\n")[-1]

print("\nModel Output:\n")
print(response)