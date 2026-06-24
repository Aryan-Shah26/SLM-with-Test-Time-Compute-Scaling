import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
from datasets import load_dataset
from tqdm import tqdm # You may need to run: pip install tqdm
from inference_engine import SLMEngine

def run_large_benchmark(num_questions=500):
    print(f"Loading SLM Engine & GSM8K Dataset ({num_questions} questions)...")
    engine = SLMEngine() # Ensure your path points to the models/ directory
    
    # Load and shuffle the test dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    subset = dataset.shuffle(seed=42).select(range(num_questions))
    
    results = []
    correct_count = 0
    
    print("\nStarting Silent Benchmark Loop...")
    print("Progress will update on the bar below. No terminal clutter.")
    print("-" * 50)
    
    # tqdm creates the progress bar
    for i, item in enumerate(tqdm(subset, desc="Evaluating Benchmark")):
        question = item['question']
        ground_truth = item['answer'].split("####")[-1].strip()
        
        prompt = engine.format_prompt(question)
        current_text = prompt
        
        # The Agentic Generation Loop
        for step in range(15):
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
                
        # Extract model's final answer safely
        final_output = current_text.split("<|im_start|>assistant")[-1]
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', final_output)
        model_answer = boxed_match.group(1).strip() if boxed_match else None
        
        # Grade the answer
        is_correct = (model_answer == ground_truth)
        if is_correct:
            correct_count += 1
            
        results.append({
            "id": i,
            "question": question,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        })
        
        # Checkpoint: Save to disk every 50 questions
        if (i + 1) % 50 == 0:
            with open("gsm8k_benchmark_results.json", "w") as f:
                json.dump({
                    "status": f"In Progress ({i+1}/{num_questions})",
                    "current_accuracy": (correct_count / (i + 1)) * 100, 
                    "results": results
                }, f, indent=4)

    # Final Math & Export
    final_accuracy = (correct_count / num_questions) * 100
    with open("gsm8k_benchmark_results.json", "w") as f:
        json.dump({
            "status": "Complete",
            "final_accuracy": final_accuracy, 
            "total_correct": correct_count, 
            "total_questions": num_questions,
            "results": results
        }, f, indent=4)
        
    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETE")
    print(f"Total Questions Evaluated: {num_questions}")
    print(f"Total Correct: {correct_count}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print("Full trace saved to: gsm8k_benchmark_results.json")
    print("=" * 50)

if __name__ == "__main__":
    # You can change this number to 300 if you are tight on time
    run_large_benchmark(num_questions=500)