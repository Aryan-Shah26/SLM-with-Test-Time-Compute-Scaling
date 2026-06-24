from unsloth import FastLanguageModel

print("Loading your SFT Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen-0.5b-thinking-lora", # Point to your SFT folder
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = False, # MUST be False to safely fuse weights
)

print("Fusing weights into a new base model...")
model.save_pretrained_merged("qwen-0.5b-sft-merged", tokenizer, save_method = "merged_16bit")
print("Merge complete! Saved to /qwen-0.5b-sft-merged")