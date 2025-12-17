# Dialogue Summarization: Fine-Tuned Transformers vs Frontier LLMs (2025)

This project answers a concrete engineering question: **what’s the most effective way to summarize noisy chat logs in 2025?**  
Instead of assuming “just send it to a frontier LLM,” I benchmarked **three approaches**—a custom encoder/decoder, purpose-built seq2seq models, and zero-shot API models—across **quality (ROUGE), latency, and cost** on the SAMSum dataset.

**Bottom line:** a **fine-tuned BART** specialist beat large frontier models on this task’s constraints—**higher ROUGE, ~3× faster**, and effectively **$0 marginal cost** at scale—while also showing why architecture choices (especially **pretrained cross-attention**) matter.

## Quick links (start here)

- **Overview Notebook (project dashboard):**  
  https://github.com/LittleHouse75/ml-capstone-summarization-models/blob/main/notebooks/00_introduction.ipynb

- **Video Presentation:**  
  https://github.com/LittleHouse75/ml-capstone-summarization-models/blob/main/reports/VideoPresentation.md

- **Pitch:**  
  https://github.com/LittleHouse75/ml-capstone-summarization-models/blob/main/reports/Pitch-Updated.md

- **Reflection:**  
  https://github.com/LittleHouse75/ml-capstone-summarization-models/blob/main/reports/Reflection-B.md

## What this project demonstrates

- **Specialization beats generalization (for this use case):** fine-tuned seq2seq models (BART/T5) outperformed zero-shot frontier LLMs on terse, outcome-focused summaries.
- **Architecture matters:** a DistilBERT + DistilGPT-2 “Frankenstein” model worked, but trained inefficiently because cross-attention was learned from scratch.
- **The “API tax” is real:** frontier LLMs were easiest to prototype with, but slower, costlier, and less consistent with the dataset’s strict brevity/style.
