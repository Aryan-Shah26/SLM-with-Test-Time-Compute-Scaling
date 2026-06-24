# Local System 2 Reasoning Engine (SLM)

An end-to-end implementation of a Test-Time Compute reasoning pipeline using a customized 0.5B parameter Small Language Model. 

This project replicates the core architectural concepts behind models like OpenAI's o1 and DeepSeek R1, optimized to run entirely locally on a 4GB VRAM edge device.

## 🚀 Architecture & Pipeline

1. **Supervised Fine-Tuning (SFT):** Fine-tuned `Qwen2.5-0.5B` using 4-bit LoRA quantization to establish strict `<think>` tag XML reasoning protocols.
2. **Reinforcement Learning (GRPO):** Implemented an automated RL sandbox utilizing Outcome Reward Models (ORM) and Process Reward Models (PRM) to penalize arithmetic hallucination.
3. **Agentic Reflexion Loop:** Engineered a multi-agent LLM-as-a-Judge inference loop. The model acts as both a 'Solver' and a 'Critic', autonomously evaluating its own intermediate reasoning steps and self-correcting mathematical errors over multiple compute iterations.

## ⚙️ Tech Stack
* **Frameworks:** PyTorch, Hugging Face Transformers, Unsloth, TRL (Transformer Reinforcement Learning)
* **Techniques:** GRPO, LoRA Quantization, Beam-Search Decoding, Multi-Agent Reflexion
* **Hardware:** NVIDIA RTX 3050 (4GB VRAM)

## 📊 Results & Architectural Insights
Benchmarked against the **GSM8K** dataset using a custom A/B testing pipeline to compare inference-time scaling techniques.

* **Baseline Performance (<20%):** The 0.5B parameter model scored under 20% across both pipelines. 
* **Insight 1 (Agentic Tooling):** While the Python intercept tool successfully eliminated arithmetic hallucinations (e.g., forcing `48/2 = 24`), it could not correct the model's fundamental reading comprehension errors when setting up the initial equations.
* **Insight 2 (Attention Collapse):** The LLM-as-a-Judge Reflexion loop proved too context-heavy for a 0.5B architecture. The model suffered from "attention collapse," losing logical coherence between its generated critique and its final `<think>` block output. 
* **Conclusion:** The pipeline successfully orchestrates Test-Time Compute, but true autonomous self-correction requires scaling the base model to at least 1.5B+ parameters to maintain context across multi-turn logic validations.
