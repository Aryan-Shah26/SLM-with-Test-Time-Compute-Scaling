import json
from inference_engine import SLMEngine

def run_reflexion_loop():
    engine = SLMEngine() # Ensure path in inference_engine.py points to models/
    
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    max_iterations = 3
    execution_trace = {
        "question": question,
        "iterations": []
    }
    
    print("\n[Reflexion Engine Started]")
    print("-" * 50)
    
    # ==========================================
    # STEP 1: INITIAL ATTEMPT
    # ==========================================
    print(f"Iteration 0: Generating initial solution...")
    solver_prompt = engine.format_prompt(question)
    current_solution = generate_full_response(engine, solver_prompt)
    
    # ==========================================
    # STEP 2: THE SELF-CORRECTION LOOP
    # ==========================================
    for i in range(1, max_iterations + 1):
        print(f"\nIteration {i}: Critic evaluating logic...")
        
        # The Evaluation Prompt (Forcing the model to act as a strict grader)
        eval_prompt = f"""<|im_start|>system
You are a ruthless, highly logical math grader. Review the provided math solution.
Check every calculation step-by-step.
If the logic and final answer are 100% mathematically correct, output EXACTLY the word "OKAY" and nothing else.
If there is a mathematical error, reading comprehension error, or hallucination, output the exact error and explain why it is wrong.<|im_end|>
<|im_start|>user
Question: {question}

Solution to evaluate:
{current_solution}<|im_end|>
<|im_start|>assistant
<think>"""

        critique = generate_full_response(engine, eval_prompt)
        
        # Log the iteration data
        execution_trace["iterations"].append({
            "iteration": i,
            "solution_attempt": current_solution,
            "critique": critique
        })
        
        # Check termination condition
        if "OKAY" in critique.upper():
            print("Critic Output: OKAY. Logic verified.")
            break
            
        print(f"Critic found an error: {critique.strip()[:500]}...")
        print(f"Iteration {i}: Solver attempting correction...")
        
        # The Correction Prompt (Injecting the error back into the solver)
        correction_prompt = engine.format_prompt(f"""Question: {question}

Your previous attempt contained an error.
Previous Attempt:
{current_solution}

Critic's Feedback:
{critique}

Please think step-by-step, fix the mathematical error, and output the final corrected answer inside \\boxed{{}}.""")

        current_solution = generate_full_response(engine, correction_prompt)
        
    print("-" * 50)
    print("\n--- Final Verified Output ---")
    print(current_solution.split("<|im_start|>assistant")[-1].strip())
    
    # Export the debate history silently
    with open("reflexion_trace.json", "w") as f:
        json.dump(execution_trace, f, indent=4)

def generate_full_response(engine, prompt):
    """Helper function to cleanly generate a full response from the model"""
    inputs = engine.tokenizer([prompt], return_tensors="pt").to(engine.model.device)
    outputs = engine.model.generate(
        inputs.input_ids,
        max_new_tokens=256,
        pad_token_id=engine.tokenizer.eos_token_id,
        do_sample=False
    )
    # Decode only the newly generated tokens
    new_text = engine.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return new_text

if __name__ == "__main__":
    run_reflexion_loop()