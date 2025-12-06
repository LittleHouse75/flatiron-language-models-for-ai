
# Dialogue Summarization: From Custom Architectures to Frontier LLMs

**An investigation into the most effective architecture for summarizing informal chat logs.**

---

## 1. Problem Statement & Business Context

**The Problem: Information Overload**
Modern work happens in chat applications (Slack, Teams, Discord). However, these platforms generate massive amounts of unstructured, noisy text. Employees spend significant time scrolling through back-and-forth messages to extract simple outcomes: decisions made, action items assigned, or meeting times agreed upon.

**The Solution**
This project builds and evaluates automated systems capable of ingesting raw, messy dialogue and outputting concise, third-person summaries.

**The Constraints**
*   **Input:** Informal, multi-speaker text with slang, typos, and non-standard grammar (SAMSum dataset).
*   **Output:** High-compression summaries (median ~75% reduction) suitable for notification previews.
*   **Production Viability:** The solution must balance accuracy (ROUGE scores) with inference latency and cost.

---

## 2. Technical Approach

We evaluated three distinct architectural paradigms to solve this problem:

### **Experiment 1: The "Frankenstein" Architecture**
*   **Model:** Custom `EncoderDecoderModel` combining **DistilBERT** (Encoder) + **DistilGPT-2** (Decoder).
*   **Hypothesis:** Can we satisfy the requirement of a BERT-family encoder and autoregressive decoder by manually connecting two models that were never trained to communicate?
*   **Result:** Functional, but inefficient training convergence due to randomly initialized cross-attention layers.

### **Experiment 2: The Specialists (Seq2Seq)**
*   **Models:** **BART** (Denoising Autoencoder) and **T5** (Text-to-Text Transfer Transformer).
*   **Hypothesis:** Do purpose-built summarization architectures with pre-trained cross-attention outperform custom assemblies?
*   **Result:** Significant performance gains in both convergence speed and final summary quality.

### **Experiment 3: Frontier LLMs (Zero-Shot)**
*   **Models:** **GPT-5 Mini**, **Gemini 2.5 Flash**, **Claude 4.5 Haiku**, **Qwen 2.5**, and **Kimi K2**.
*   **Hypothesis:** Can massive zero-shot models accessing the world's knowledge beat smaller, fine-tuned local models?
*   **Result:** High fluency, but often failed to match the specific terse style required by the dataset, leading to lower ROUGE scores.

---

## 3. Results & Evaluation

We evaluated all models on the **SAMSum test set** (819 examples) using ROUGE metrics and inference latency.

### Key Findings

1.  **Fine-Tuning Wins on Quality:** The **BART** model achieved the highest ROUGE-L score (**42.13**), beating the best Frontier API model (Gemini 2.5 Flash at **35.49**) by over 6 points.
2.  **Speed:** Local models were significantly faster. BART averaged **0.2s** per summary, while the fastest API (Gemini) averaged **0.65s**, with some APIs lagging to 2.0s+.
3.  **The "Intelligence" Trap:** Frontier LLMs were "smarter" but less obedient. They tended to add conversational filler or helpful context that, while correct, penalized them against the ground truth summaries which favor extreme brevity.

| Model Category | Best Model | ROUGE-L | Latency (mean) | Cost at Scale |
| :--- | :--- | :--- | :--- | :--- |
| **Fine-Tuned Local** | **BART** | **42.13** | **~0.20s** | **~$0 (Compute)** |
| Fine-Tuned Local | T5-Small | 39.08 | ~0.23s | ~$0 (Compute) |
| Frontier API | Gemini 2.5 Flash | 35.49 | ~0.65s | High (Per Token) |
| Custom Custom | Bert+GPT2 | 30.29 | ~0.40s | ~$0 (Compute) |

---

## 4. Repository Structure

This project is organized as a series of sequential notebooks.

```text
.
├── README.md                           # This report
├── notebooks/                          # Core analysis and experiments
│   ├── 00_introduction.ipynb           # Project overview and navigation
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis of SAMSum
│   ├── 02_experiment1_bert_gpt2.ipynb  # Custom Encoder-Decoder implementation
│   ├── 03_experiment2_bart_t5.ipynb    # BART and T5 Fine-tuning
│   ├── 04_experiment3_api_models.ipynb # API benchmarking (OpenRouter)
│   └── 05_evaluation_and_conclusions.ipynb # Final cross-model analysis
├── reports/
│   └── Pitch-Updated.md                # Evolution of project scope
├── src/                                # Helper modules
│   ├── data/
│   │   └── load_data.py                # Dataset loading utilities
│   └── eval/
│       └── qualitative.py              # Qualitative sample generation
├── models/                             # [Generated] Stores trained weights & logs
└── experiments/                        # [Generated] Stores comparison plots
```

---

## 5. Reproduction Instructions

Because the fine-tuned models are too large for Git, they are **not checked into the repository**. You must reproduce them locally by running the notebooks.

### Prerequisites
*   Python 3.10+
*   A GPU is highly recommended (Training code is optimized for CUDA/ROCm).
*   **OpenRouter API Key** (Required only for notebook `04`).

### Setup
```bash
# Clone the repository
git clone <repo_url>
cd <repo_name>

# Install dependencies (assuming standard DS stack + transformers)
pip install pandas numpy matplotlib torch transformers datasets evaluate rouge_score requests
```

### Reproducing the Models
To regenerate the model artifacts (weights, logs, predictions), you must run the notebooks in order.

1.  **Open `notebooks/02_experiment1_bert_gpt2.ipynb`**
    *   Set the flag `RUN_TRAINING = True` in the configuration cell.
    *   Run all cells. This will train the DistilBERT+GPT2 model and save artifacts to `models/bert-gpt2-distil/`.

2.  **Open `notebooks/03_experiment2_bart_t5.ipynb`**
    *   Set `RUN_TRAINING_BART = True` and `RUN_TRAINING_T5 = True`.
    *   Run all cells. This will train both models and save artifacts to `models/bart/` and `models/t5/`.

3.  **Open `notebooks/04_experiment3_api_models.ipynb`**
    *   Set `RUN_API_CALLS = True`.
    *   Start with `EVALUATION_MODE = "test"` (small sample) to verify your API key, then switch to `"full"` for the complete test set.
    *   Run all cells to generate API predictions in `models/api-frontier/`.

4.  **Final Analysis:**
    *   Run `notebooks/05_evaluation_and_conclusions.ipynb` to generate the final comparison charts based on the artifacts created in steps 1-3.

---

## 6. Limitations & Future Work

**Limitations**
*   **Dataset Bias:** SAMSum focuses on casual, English-speaking, two-person dialogue. Performance may degrade on multi-party technical meetings or formal customer support threads.
*   **Tail Latency:** While API models had decent average speed, their P95 latency (worst 5%) was often 5x-10x slower than local models, posing a risk for real-time applications.

**Future Work**
*   **Quantization:** Attempting to run 4-bit quantized versions of Llama-3 or Mistral locally to see if we can bridge the gap between "dumb but fast" (BART) and "smart but slow" (GPT-4).
*   **Human Evaluation:** ROUGE scores penalize synonyms. A human evaluation loop is needed to determine if the "verbose" API summaries are actually more useful to users despite lower metrics.[REDACTED]