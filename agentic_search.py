import re
import json
import torch
from inference_engine import SLMEngine

def run_silent_agentic_loop():
    engine = SLMEngine() # Ensure this path points to your 'models/' directory
    
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    prompt = engine.format_prompt(question)
    current_text = prompt

    # 1. Initialize the Data Structure
    execution_trace = {
        "question": question,
        "events": [], # This will store chronological logs
        "final_output": None
    }

    for step in range(15): 
        inputs = engine.tokenizer([current_text], return_tensors="pt").to(engine.model.device)
        outputs = engine.model.generate(
            inputs.input_ids,
            max_new_tokens=30, 
            pad_token_id=engine.tokenizer.eos_token_id,
            do_sample=False
        )
        
        new_text = engine.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        math_match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)?', new_text)
        
        if math_match:
            num1, op, num2, hallucinated_ans = math_match.groups()
            try:
                real_ans = eval(f"{num1} {op} {num2}")
                if real_ans == int(real_ans): real_ans = int(real_ans)
                corrected_equation = f"{num1} {op} {num2} = {real_ans}"
                
                # 2. Log the Tool Intercept instead of printing
                execution_trace["events"].append({
                    "type": "tool_intercept",
                    "step": step,
                    "attempted_math": f"{num1} {op} {num2} = {hallucinated_ans}",
                    "forced_correction": corrected_equation
                })
                
                parts = re.split(r'\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?\s*=\s*(?:\d+(?:\.\d+)?)?', new_text, 1)
                safe_new_text = parts[0] + corrected_equation
                current_text += safe_new_text + "\n" 
                continue 
            except Exception:
                pass 
                
        current_text += new_text
        
        # 3. Log the standard generation chunks
        execution_trace["events"].append({
            "type": "generation_chunk",
            "step": step,
            "content": new_text
        })
        
        if "</think>" in new_text or "\\boxed" in new_text:
            break

    # 4. Finalize the trace
    final_output = current_text.split("<|im_start|>assistant")[-1].strip()
    execution_trace["final_output"] = final_output

    # 5. Export to JSON (Silent Output)
    with open("agent_trace.json", "w") as f:
        json.dump(execution_trace, f, indent=4)
        
    return execution_trace

if __name__ == "__main__":
    run_silent_agentic_loop()