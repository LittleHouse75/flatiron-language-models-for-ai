Dialogue Summarization with Transformer Models

Proof-of-Concept for Automated Chat Summaries
*Flatiron School – Final Capstone*

### Overview

Modern chat platforms bombard users with long, chaotic message threads. Important decisions and action items get buried in noise. This project builds a proof-of-concept summarization system that turns raw dialogues into short, accurate summaries.

The work follows a full data-science workflow:
data exploration → model design → fine-tuning → evaluation → comparison → conclusions.

We evaluate three modeling strategies:
1. **Experiment 1 – Custom Encoder–Decoder:**

⠀BERT encoder → GPT-2 decoder (cross-attention added via HuggingFace EncoderDecoderModel).
2. **Experiment 2 – Purpose-Built Seq2Seq:**

⠀BART and T5, pretrained for summarization.
3. **Experiment 3 – Frontier LLMs via API:**

⠀Zero-shot / instructed summarizers (GPT-4o-mini, Claude, etc.).

The dataset is **SAMSum**, a human-annotated messenger-style conversation corpus.

⸻

### Key Features
* End-to-end pipeline for dialogue summarization
* Reusable preprocessing and training utilities (src/)
* ROUGE-based evaluation across all model families
* Side-by-side qualitative comparison
* Saved checkpoints for reproducibility
* Final report and video presentation delivered per flatiron requirements

⠀
⸻

### Project Structure

project-dialogue-summarization/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                # direct pulls from SAMSum (unchanged)
│   ├── interim/            # tokenized HF datasets / cached splits
│   └── processed/          # final tensors ready for training
│
├── notebooks/
│   ├── 00_intro_and_setup.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_experiment1_bert_gpt2.ipynb
│   ├── 03_experiment2_bart_t5.ipynb
│   ├── 04_experiment3_api_models.ipynb
│   ├── 05_evaluation_and_comparison.ipynb
│   └── 06_conclusions.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py           # dataset loading + splitting
│   │   └── preprocess.py          # tokenization / truncation logic
│   │
│   ├── models/
│   │   ├── build_bert_gpt2.py     # encoder-decoder glue model
│   │   ├── build_bart.py
│   │   └── build_t5.py
│   │
│   ├── train/
│   │   ├── trainer_seq2seq.py     # training + eval loop
│   │   └── callbacks.py           # early stopping / checkpointing
│   │
│   ├── eval/
│   │   ├── rouge_eval.py
│   │   └── qualitative.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── config.py
│       └── prompt_utils.py        # used for Experiment 3
│
├── models/
│   ├── bert-gpt2/                 # exp1 checkpoints
│   ├── bart/                      # exp2 checkpoints
│   ├── t5/
│   └── frontier_llm/              # prompt templates, cached outputs
│
├── experiments/
│   ├── exp1_bert_gpt2_results/
│   ├── exp2_bart_results/
│   └── exp3_api_llm_results/
│
└── reports/
    ├── pitch.pdf
    ├── final_report.pdf
    └── video_slides/


⸻

### How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the notebooks

The main workflow lives in:
* notebooks/00_intro_and_setup.ipynb
* …then progress sequentially through

⠀EDA → Exp1 → Exp2 → Exp3 → Evaluation → Conclusions

### 3. Model weights

Fine-tuned model weights are stored under:

models/<model-name>/

Use HuggingFace’s from_pretrained() to load them.

⸻

### Results (Short Summary)

*(You’ll expand this section after training finishes.)*
* BERT→GPT-2 works but trains slowly because cross-attention is randomly initialized.
* BART/T5 produce stronger ROUGE scores with significantly less compute.
* Frontier LLMs produce the best summaries but with real cost/latency trade-offs.
* The final comparison notebook shows quality, latency, and cost across all three.

⠀
⸻

### Deliverables
* Jupyter notebooks with full pipeline
* Saved checkpoints for all models
* Evaluation metrics + qualitative analysis
* Final 3–5 page PDF report
* 7–10 minute video walkthrough
* This repository README

⠀
⸻

### License / Notes

This project is for educational use within the Flatiron School AI/ML program.
SAMSum dataset accessed via HuggingFace Datasets.pp