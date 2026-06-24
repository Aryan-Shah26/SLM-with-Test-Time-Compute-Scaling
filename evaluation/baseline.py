from inference_engine import SLMEngine
import time

def run_baseline():
    engine = SLMEngine()
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

    print("\n--Running Greedy Baseline--")
    start_time = time.time()
    outputs = engine.generate(engine.format_prompt(question), do_sample=False, temperature=0.0, max_new_tokens=512, num_return_sequences=1)
    end_time = time.time()

    response = outputs[0]
    final_answer = engine.extract_answer(response)

    print(f"Reasoning Path:\n{response}")
    print(f"\nExtracted Answer: {final_answer}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_baseline()