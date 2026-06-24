import torch
import re
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


# =================================================================
# 1. Configuration & Model Loading
# =================================================================
max_seq_length = 2048 # Keep this relatively low for 16GB RAM / 3050 VRAM limits
load_in_4bit = True   # Mandatory for your hardware

print("Loading Qwen 0.5B with Unsloth in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)

# =================================================================
# 2. Injecting the LoRA Adapters
# =================================================================
# This tells the model which specific neural layers we are allowed to train
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # The size of the adapter. 16 is a great balance of speed and learning capability.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", # Massive VRAM saver
    random_state = 3407,
)

# =================================================================
# 3. Dataset Formatting
# =================================================================
# NOTE: You will need to map your dataset columns here!
def reformat_gsm8k(raw_answer):
    """Translates GSM8K's raw format into our <think> \boxed{} format."""
    if "####" in raw_answer:
        parts = raw_answer.split("####")
        reasoning = parts[0].strip()
        
        # Remove the weird <<math>> annotations GSM8K uses
        reasoning = re.sub(r'<<.*?>>', '', reasoning)
        
        answer = parts[1].strip()
        return f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
    return raw_answer

def formatting_prompts_func(examples):
    instructions = examples["question"] 
    outputs      = examples["answer"]  
    texts = []
    
    for instruction, output in zip(instructions, outputs):
        # Reformat the target output BEFORE we train the model on it
        formatted_output = reformat_gsm8k(output)
        
        messages = [
            {"role": "system", "content": "You are a logical math solver. Think step-by-step inside <think> tags, then output the final numerical answer inside \\boxed{}."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": formatted_output} # Now it sees the correct tags!
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }

# Load your local dataset (Assumes JSONL or CSV)
# Change "your_dataset.jsonl" to your actual file path
print("Loading dataset...")
dataset = load_dataset("parquet", data_files="gsm8k/main/train-00000-of-00001.parquet", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# =================================================================
# 4. The Trainer Engine
# =================================================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, 
    args = SFTConfig( # <-- Updated to SFTConfig
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4, 
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", 
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# =================================================================
# 5. Execute Training & Save
# =================================================================
print("Starting SFT Training...")
trainer_stats = trainer.train()

print("Training complete! Saving LoRA adapters...")
model.save_pretrained("qwen-0.5b-thinking-lora")
tokenizer.save_pretrained("qwen-0.5b-thinking-lora")
print("Saved to /qwen-0.5b-thinking-lora")