import torch
import time
from inference_engine import SLMEngine

def heuristic_step_scorer(step_text):
    """
    A lightweight verifier/critic function.
    In a full system, this would be a reward model or an execution sandbox.
    For now, we penalize empty steps, repetitive phrases, or clear gibberish.
    """

    score = 1.0
    stripped = step_text.strip()

    if not stripped:
        return 0.0

    # Reward structural markers that show structured thinking
    if "So," in stripped or "Therefore," in stripped:
        score += 0.2
    if any(op in stripped for op in ["+", "-", "*", "/"]):
        score += 0.3  # Favor steps explicitly calculating things
        
    # Penalize loops or too short steps
    if len(stripped) < 5:
        score -= 0.5
        
    return max(0.1, score)

def run_step_level_search(beam_width=3, max_steps=5):
    engine = SLMEngine()
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    print(f"\n--- Running Step-Level Beam Search (Beam Width = {beam_width}) ---")
    start_time = time.time()

    # Every active beam is a dictionary tracking its current text, aggregate score, and completeness
    beams = [{"text": engine.format_prompt(question), "score": 1.0, "done": False}]

    for step in range(max_steps):
        print(f"\n--- Processing Reasoning Step {step + 1}/{max_steps} ---")
        new_candidate_beams = []
        all_done = True

        for beam in beams:
            if beam["done"]:
                new_candidate_beams.append(beam)
                continue
            
            all_done = False
            
            # Generate multiple alternative next-steps for this specific beam branch
            # We use a stop token configuration to halt generation at a newline
            inputs = engine.tokenizer(beam["text"], return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                # We sample N candidates for the *next single step*
                outputs = engine.model.generate(
                    **inputs,
                    max_new_tokens=64, # Small chunk size per step
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=beam_width, 
                    pad_token_id=engine.tokenizer.eos_token_id
                )

            for output in outputs:
                generated_tokens = output[input_length:]
                decoded_text = engine.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Split by newline to capture just the immediate reasoning step
                step_lines = decoded_text.split('\n')
                immediate_step = step_lines[0] if step_lines else decoded_text
                
                # Evaluate the quality of this step
                step_score = heuristic_step_scorer(immediate_step)
                
                # Construct the updated path
                new_text = beam["text"] + "\n" + immediate_step
                is_done = "\\boxed" in immediate_step or engine.tokenizer.eos_token in decoded_text
                
                new_candidate_beams.append({
                    "text": new_text,
                    "score": beam["score"] * step_score, # Cumulative step score
                    "done": is_done
                })

        if all_done:
            break

        # PRUNING STEP: Sort all candidate branches globally and keep the top K (beam_width)
        new_candidate_beams.sort(key=lambda x: x["score"], reverse=True)
        beams = new_candidate_beams[:beam_width]

        for idx, b in enumerate(beams):
            print(f"  Top Branch {idx+1} [Score: {b['score']:.2f}]: ... {b['text'][-60:].strip()}")

    # Final extraction from the best scoring beam
    best_beam = beams[0]
    final_answer = engine.extract_answer(best_beam["text"])
    
    end_time = time.time()
    print("\n--- Final Search Result ---")
    print(f"Optimal Path Cleared:\n{best_beam['text']}")
    print(f"\nExtracted Answer from Best Branch: {final_answer}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_step_level_search(beam_width=3, max_steps=6)