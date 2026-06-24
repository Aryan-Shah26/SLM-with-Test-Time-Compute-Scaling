from inference_engine import SLMEngine
from collections import Counter
import time

def run_best_of_n(n=5):
    engine = SLMEngine()

    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    prompt = engine.format_prompt(question)
    
    print(f"\n--- Running Best-of-{n} (Self-Consistency) ---")
    start_time = time.time()

    all_answers = []

    for i in range(n):
        print(f"Generating reasoning path {i+1}/{n}...")
        outputs = engine.generate(prompt, do_sample=True, temperature=0.7, max_new_tokens=512, num_return_sequences=1)

        response = outputs[0]
        extracted = engine.extract_answer(response)

        if extracted:
            all_answers.append(extracted)

    end_time = time.time()

    if all_answers:
        vote_counts = Counter(all_answers)
        best_answer, count = vote_counts.most_common(1)[0]

        print("\n--- Results ---")
        print(f"All extracted answers: {all_answers}")
        print(f"Final Consensus Answer: {best_answer} (with {count}/{n} votes)")
    else:
        print("\nFailed to extract any properly formatted answers.")
        
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_best_of_n(n=5)