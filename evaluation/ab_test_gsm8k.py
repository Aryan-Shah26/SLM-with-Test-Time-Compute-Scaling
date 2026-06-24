import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
from datasets import load_dataset
from tqdm import tqdm
from inference_engine import SLMEngine

def extract_answer(text):
    """Safely extracts the boxed answer."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    return match.group(1).strip() if match else None

def run_tool_intercept(engine, question):
    """Runs the Python Calculator Loop."""
    prompt = engine.format_prompt(question)
    current_text = prompt
    for _ in range(10): # Shorter loop for speed
        inputs = engine.tokenizer([current_text], return_tensors="pt").to(engine.model.device)
        outputs = engine.model.generate(
            inputs.input_ids, max_new_tokens=30, 
            pad_token_id=engine.tokenizer.eos_token_id, do_sample=False
        )
        new_text = engine.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        math_match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)?', new_text)
        if math_match:
            num1, op, num2, _ = math_match.groups()
            try:
                real_ans = eval(f"{num1} {op} {num2}")
                if real_ans == int(real_ans): real_ans = int(real_ans)
                corrected_equation = f"{num1} {op} {num2} = {real_ans}"
                parts = re.split(r'\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?\s*=\s*(?:\d+(?:\.\d+)?)?', new_text, 1)
                new_text = parts[0] + corrected_equation
            except Exception:
                pass
                
        current_text += new_text + "\n"
        if "</think>" in new_text or "\\boxed" in new_text:
            break
            
    final_output = current_text.split("<|im_start|>assistant")[-1]
    return extract_answer(final_output)

def run_reflexion(engine, question):
    """Runs the LLM-as-a-Judge Loop."""
    # 1. Initial Attempt
    prompt = engine.format_prompt(question)
    inputs = engine.tokenizer([prompt], return_tensors="pt").to(engine.model.device)
    outputs = engine.model.generate(inputs.input_ids, max_new_tokens=150, pad_token_id=engine.tokenizer.eos_token_id, do_sample=False)
    current_solution = engine.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 2. Reflexion Loop
    for _ in range(2): # Max 2 corrections to save time
        eval_prompt = f"""<|im_start|>system
You are a math grader. Check the solution. If 100% correct, output exactly "OKAY". If wrong, explain the error.<|im_end|>
<|im_start|>user
Question: {question}
Solution: {current_solution}<|im_end|>
<|im_start|>assistant
<think>"""
        eval_inputs = engine.tokenizer([eval_prompt], return_tensors="pt").to(engine.model.device)
        eval_outputs = engine.model.generate(eval_inputs.input_ids, max_new_tokens=100, pad_token_id=engine.tokenizer.eos_token_id, do_sample=False)
        critique = engine.tokenizer.decode(eval_outputs[0][eval_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        if "OKAY" in critique.upper():
            break
            
        correction_prompt = engine.format_prompt(f"Question: {question}\nPrevious Error: {critique}\nFix it and output \\boxed{{}}.")
        corr_inputs = engine.tokenizer([correction_prompt], return_tensors="pt").to(engine.model.device)
        corr_outputs = engine.model.generate(corr_inputs.input_ids, max_new_tokens=150, pad_token_id=engine.tokenizer.eos_token_id, do_sample=False)
        current_solution = engine.tokenizer.decode(corr_outputs[0][corr_inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return extract_answer(current_solution)

def run_ab_test(num_questions=50):
    print("Loading Engine & Preparing A/B Test...")
    engine = SLMEngine()
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    subset = dataset.shuffle(seed=42).select(range(num_questions))
    
    results = []
    tool_correct = 0
    reflexion_correct = 0
    
    for i, item in enumerate(tqdm(subset, desc="A/B Testing Pipelines")):
        question = item['question']
        ground_truth = item['answer'].split("####")[-1].strip()
        
        # Pipeline A: Tool Intercept
        tool_ans = run_tool_intercept(engine, question)
        if tool_ans == ground_truth: tool_correct += 1
            
        # Pipeline B: Reflexion
        reflexion_ans = run_reflexion(engine, question)
        if reflexion_ans == ground_truth: reflexion_correct += 1
            
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "tool_answer": tool_ans,
            "reflexion_answer": reflexion_ans
        })
        
        # Save aggressively
        with open("ab_test_results.json", "w") as f:
            json.dump({
                "questions_evaluated": i + 1,
                "tool_accuracy": (tool_correct / (i + 1)) * 100,
                "reflexion_accuracy": (reflexion_correct / (i + 1)) * 100,
                "raw_data": results
            }, f, indent=4)

    print("\n" + "=" * 50)
    print("A/B TEST COMPLETE")
    print(f"Tool-Intercept Accuracy:  {(tool_correct / num_questions) * 100:.2f}%")
    print(f"Reflexion Loop Accuracy:  {(reflexion_correct / num_questions) * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    run_ab_test(num_questions=50) # Keep at 50 to finish within 15-20 minutes