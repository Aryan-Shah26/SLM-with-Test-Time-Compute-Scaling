import torch
import time
import re
from transformers import StoppingCriteria, StoppingCriteriaList
from inference_engine import SLMEngine

# ---------------------------------------------------------
# 1. Custom Stopping Criteria
# ---------------------------------------------------------
class StepStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Halt generation if the last generated token is a newline
        last_token = input_ids[0][-1].item()
        return last_token in self.stop_token_ids

# ---------------------------------------------------------
# 2. The Math Critic (Verifier)
# ---------------------------------------------------------
def math_critic_scorer(step_text):
    """
    Acts as a verifier during Test-Time Compute.
    Extracts basic arithmetic from the step and verifies it using Python.
    """
    score = 1.0
    
    # Penalize empty steps
    if not step_text.strip():
        return 0.1

    # Look for explicit equations in the format: Number Operator Number = Number
    # e.g., "48 / 2 = 24"
    equations = re.findall(r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', step_text)
    
    for eq in equations:
        num1, op, num2, result = eq
        try:
            num1, num2, result = float(num1), float(num2), float(result)
            
            # Verify the math
            is_correct = False
            if op == '+' and num1 + num2 == result: is_correct = True
            elif op == '-' and num1 - num2 == result: is_correct = True
            elif op == '*' and num1 * num2 == result: is_correct = True
            elif op == '/' and num2 != 0 and num1 / num2 == result: is_correct = True
            
            if is_correct:
                score += 1.5  # Reward verified, correct intermediate math
    
        except ValueError:
            pass

    # Reward finding the final answer format
    if "\\boxed{" in step_text:
        score += 2.0
        
    return max(0.01, score) # Ensure score doesn't go below 0

# ---------------------------------------------------------
# 3. The Guided Search Loop
# ---------------------------------------------------------
def run_guided_search(beam_width=3, max_steps=6):
    engine = SLMEngine()
    
    # Get the token ID for a newline to feed our stopping criteria
    newline_token_ids = engine.tokenizer.encode('\n', add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([StepStoppingCriteria(newline_token_ids)])

    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    print(f"\n--- Running Guided Math Search (Beam Width = {beam_width}) ---")
    start_time = time.time()

    beams = [{"text": engine.format_prompt(question), "score": 1.0, "done": False}]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        new_candidate_beams = []
        all_done = True

        for beam in beams:
            if beam["done"]:
                new_candidate_beams.append(beam)
                continue
            
            all_done = False
            inputs = engine.tokenizer(beam["text"], return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = engine.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.4, # Slightly higher temp to encourage diverse reasoning paths
                    num_return_sequences=beam_width, 
                    pad_token_id=engine.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria # Inject our custom stop rule
                )

            for output in outputs:
                generated_tokens = output[input_length:]
                immediate_step = engine.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Run the math critic
                step_score = math_critic_scorer(immediate_step)
                
                new_text = beam["text"] + immediate_step + "\n"
                is_done = "\\boxed" in immediate_step or engine.tokenizer.eos_token in immediate_step
                
                new_candidate_beams.append({
                    "text": new_text,
                    "score": beam["score"] * step_score,
                    "done": is_done
                })

        if all_done: break

        # Sort and prune based on the Math Critic's scores
        new_candidate_beams.sort(key=lambda x: x["score"], reverse=True)
        beams = new_candidate_beams[:beam_width]

        for idx, b in enumerate(beams):
            # Print a snippet to watch the pruning in real-time
            snippet = b['text'].strip().split('\n')[-1]
            print(f"  Branch {idx+1} [Score: {b['score']:.2f}]: {snippet}")

    best_beam = beams[0]
    final_answer = engine.extract_answer(best_beam["text"])
    
    print("\n--- Final Output ---")
    print(f"Optimal Verified Path:\n{best_beam['text']}")
    print(f"\nExtracted Answer: {final_answer}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_guided_search(beam_width=3, max_steps=8)