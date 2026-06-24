import torch
import re
from unsloth import FastLanguageModel

class SLMEngine:
    def __init__(self, model_name = "models/grpo_outputs/checkpoint-100"):
        print(f"Loading custom Unsloth model: {model_name} in 4-bit...")

        self.max_seq_length = 2048
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # Loads your newly trained folder
            max_seq_length = self.max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        
        # Enable Unsloth's 2x faster inference magic
        FastLanguageModel.for_inference(self.model)
        print("Thinking Model loaded successfully.")

    def format_prompt(self, question):
        """Formats the prompt using our training template."""
        messages = [
            {"role": "system", "content": "You are a logical math solver. Think step-by-step inside <think> tags, then output the final numerical answer inside \\boxed{}."},
            {"role": "user", "content": question}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def extract_answer(self, text):
        """Extracts the final answer from the \boxed{} format."""
        match = re.search(r'\\boxed\{(.*?)\}', text)
        if match:
            return match.group(1).strip()
        
        # Fallback
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            return numbers[-1]
        return None
    
    def generate(self, prompt, do_sample = False, temperature = 0.0, max_new_tokens = 512, num_return_sequences = 1, stopping_criteria = None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = do_sample,
            temperature = temperature,
            num_return_sequences = num_return_sequences,
            pad_token_id = self.tokenizer.eos_token_id,
            stopping_criteria = stopping_criteria
        )

        input_length = inputs.input_ids.shape[1]
        response = []
        for output in outputs:
            response.append(self.tokenizer.decode(output[input_length:], skip_special_tokens=True))
        return response