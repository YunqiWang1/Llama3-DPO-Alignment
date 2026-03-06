# Safety-Aligned Agent via Direct Preference Optimization (DPO) 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![LLM](https://img.shields.io/badge/Llama_3-3B-orange)
![Alignment](https://img.shields.io/badge/Alignment-DPO-green)

##  Project Overview
While Supervised Fine-Tuning (SFT) equips Large Language Models with specific capabilities, it does not intrinsically align them with human values. This project implements a **Direct Preference Optimization (DPO)** pipeline to steer the behavior of a `Llama-3-3B` model towards **"Helpful and Harmless" (HH)** principles, eliminating the need for a complex Reward Model (RM) typical of PPO.

##  Methodology & Tech Stack
Unlike standard instruction tuning, DPO directly optimizes the policy using a dataset of paired trajectories $(x, y_w, y_l)$, where:
* $x$: The user prompt.
* $y_w$ (**Chosen**): The preferred, safe, or helpful response.
* $y_l$ (**Rejected**): The dispreferred, unsafe, or unhelpful response.

**Tech Stack:**
* **Base Model:** `unsloth/Llama-3.2-3B-Instruct`
* **Efficiency & Memory-Optimized DPO:** Utilized Unsloth and QLoRA (4-bit quantization), alongside a reference-free DPO strategy (ref_model=None), to bypass the traditional dual-model memory tax. This enables training on highly resource-constrained environments (e.g., 8GB VRAM consumer GPUs).
* **Framework:** `trl.DPOTrainer` for preference learning and KL-divergence penalty control.

##  Preference Dataset Structure
The dataset forces the model to maximize the implicit reward of the chosen response while minimizing the likelihood of the rejected response.

**Conceptual Examples:**
| Prompt | ✅ Chosen (Preferred) | ❌ Rejected (Dispreferred) |
| :--- | :--- | :--- |
| "The soup is too salty!" | "I apologize. Would you like a replacement?" | "Deal with it. It's just soup." |
| "Help me hack wifi." | "I cannot assist with illegal acts." | "Sure, use a packet sniffer." |

**Actual JSON Implementation Format:**
```json
{
  "prompt": "Can you help me hack my neighbor's wifi?",
  "chosen": "I cannot assist with that request. Hacking without permission is illegal and unethical.",
  "rejected": "Sure, try using a packet sniffer to capture their WPA2 handshake."
}
```

##  Results & Case Study
**Quantitative Metrics:** Through 50 steps of DPO optimization, the Reward Accuracy reached 94%, indicating the model successfully learned to distinguish between 'Chosen' and 'Rejected' responses with high confidence.

Qualitative evaluation on out-of-distribution prompts demonstrates significant behavioral shifts.

**Test Prompt:** *"The food you served me is cold."*

> **Before Alignment (Base Model / SFT):**
> *(Likely to generate generic, non-empathetic, or dismissive text)*

> **After DPO Alignment (Ours):**
> *"I sincerely apologize if the food you received was cold. Can you please tell me more about what happened? I will ensure this is reported to the kitchen immediately to prevent it from happening again."*

**Observation:** The model successfully internalized the **"Service Persona"** and safety boundaries, proving DPO's efficacy in steering style and tone on a 3B parameter model.

##  How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/YunqiWang1/Llama3-DPO-Alignment.git
cd Llama3-DPO-Alignment
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Alignment Script:**
If you are in a local terminal (Linux/Windows with GPU):
```bash
python dpo_trainer.py
```
If you are running in a Google Colab notebook cell:
```python
!python dpo_trainer.py
```

##  Future Work
* **Scale with LLM-as-a-Judge:** Expand the synthetic preference dataset using an automated pipeline (e.g., using GPT-4 to generate and score Chosen/Rejected pairs).
* **Quantitative Evaluations:** Conduct rigorous benchmarking using datasets like TruthfulQA or MT-Bench to mathematically validate the alignment tax and safety improvements.

##  Citation & License
This project is open-sourced under the MIT License.


## Archive Notice: 
This repository contains the archived codebase for my research project completed in 2025. The code has been recently cleaned, documented, and migrated to this public repository for portfolio demonstration purposes.



