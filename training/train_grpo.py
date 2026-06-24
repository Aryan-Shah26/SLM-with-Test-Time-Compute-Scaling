import torch
import re
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig

# =================================================================
# 1. Configuration & Model Loading (Base Model)
# =================================================================
max_seq_length = 1024 # Reduced to leave room for generation during training
print("Loading Base Model for RL...")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen-0.5b-sft-merged", # <-- Point to your new fused model here!
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True, # Back to 4-bit for training
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# =================================================================
# 2. Dataset Prep (Questions and True Answers)
# =================================================================
def extract_gsm8k_answer(raw_answer):
    """Extracts the final pure number from GSM8K's raw target."""
    if "####" in raw_answer:
        return raw_answer.split("####")[1].strip()
    return raw_answer.strip()

def prep_dataset(example):
    question = example["question"]
    # We set up the prompt, but we DO NOT provide the formatted answer.
    # The model has to figure it out itself during generation.
    messages = [
        {"role": "system", "content": "You are a logical math solver. Think step-by-step inside <think> tags, then output the final numerical answer inside \\boxed{}."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    answer = extract_gsm8k_answer(example["answer"])
    return {"prompt": prompt, "answer": answer}

print("Loading Dataset...")
dataset = load_dataset("parquet", data_files="gsm8k/main/train-00000-of-00001.parquet", split="train")
# Use a tiny subset for our test run to see it learn quickly
dataset = dataset.select(range(500)).map(prep_dataset) 

# =================================================================
# 3. The Reward Functions (The Critics)
# =================================================================
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}
if hasattr(model, "base_model"):
    model.base_model.warnings_issued = {}
def logic_reward_func(completions, **kwargs):
    """Rewards mathematically correct intermediate equations, punishes bad math."""
    rewards = []
    for comp in completions:
        text = comp[0]['content'] if isinstance(comp, list) else str(comp)
        
        # Extract only the thinking block
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue
            
        think_content = think_match.group(1)
        
        # Find basic equations like "A + B = C" or "A * B = C"
        equations = re.findall(r'(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)', think_content)
        
        if not equations:
            rewards.append(0.0) # No math shown
            continue
            
        step_reward = 0.0
        for num1, op, num2, result in equations:
            try:
                # Use Python to evaluate the left side of the equation
                expected = eval(f"{num1} {op} {num2}")
                if float(expected) == float(result):
                    step_reward += 0.5 # Good math step!
                else:
                    step_reward -= 1.0 # Hallucinated math! Punish severely.
            except:
                pass
                
        rewards.append(step_reward)
    return rewards

def accuracy_reward_func(completions, answer, **kwargs):
    """Reward +2.0 if the math is actually correct."""
    rewards = []
    for comp, ans in zip(completions, answer):
        text = comp[0]['content'] if isinstance(comp, list) else str(comp)
        
        match = re.search(r'\\boxed\{(.*?)\}', text)
        if match and match.group(1).strip() == str(ans).strip():
            rewards.append(2.0)
        else:
            rewards.append(0.0) # Wrong answer gets nothing
    return rewards

# =================================================================
# 4. The GRPO Trainer
# =================================================================
training_args = GRPOConfig(
    output_dir = "grpo_outputs",
    learning_rate = 1e-5, # RL uses a much smaller learning rate than SFT
    per_device_train_batch_size = 1, # Strict constraint for 3050
    gradient_accumulation_steps = 4,
    max_prompt_length = 256,
    max_completion_length = 512,
    num_generations = 4, # The 'Group' size. 4 paths per question. 
    max_steps = 100,
    logging_steps = 1,
    optim = "adamw_8bit",
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [logic_reward_func, accuracy_reward_func],
    args = training_args,
    train_dataset = dataset,
)

# =================================================================
# 5. Execute Training & Save
# =================================================================
print("Starting GRPO Reinforcement Learning...")
trainer.train()

print("Training complete! Saving GRPO LoRA adapters...")
model.save_pretrained("qwen-0.5b-grpo-reasoner")
tokenizer.save_pretrained("qwen-0.5b-grpo-reasoner")
print("Saved to /qwen-0.5b-grpo-reasoner")