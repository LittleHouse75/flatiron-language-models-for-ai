Tim Nevits
Flatiron School
AI / ML Program

# Capstone 3 Self Reflection (500-750 Words)


## **Accomplishments**

Looking back over the project, the through-line that stands out is the sheer breadth of what I attempted and how much of it actually worked. Technically, the biggest success — and the one that required the most stubbornness — was getting the custom DistilBERT → DistilGPT-2 encoder-decoder model into respectable shape. This architecture was always going to be awkward, because the encoder and decoder were never meant to talk to each other, and the cross-attention layers begin completely untrained. Early on, the model was flailing, and I started wondering whether this part of the project was going to collapse under its own weight. The breakthrough came when I discovered that the repeated-n-gram penalty in my generation config was quietly strangling output quality. Once I removed it, ROUGE scores jumped by roughly 10 points, and the model finally behaved like a legitimate baseline instead of a research artifact.

Another accomplishment was the overall scope. I didn’t settle for building one model and calling it done. I trained three architectures — the Frankenstein model, BART, and T5 — and compared them against five API-based frontier models. Eight models altogether. That breadth let me build a real tradeoff analysis between custom fine-tuning, traditional seq2seq models, and modern LLM APIs. It also forced me to think about the entire workflow: dataset understanding, preprocessing, architecture choices, training dynamics, evaluation, and final comparison. Process-wise, keeping all of that coherent was an achievement in itself.

## **Opportunity for Growth**

The most significant lesson learned came from letting LLMs over-steer my decision-making early in the project. Claude recommended several architectural and training changes that sounded reasonable but ultimately degraded performance. I spent nearly two days following those suggestions before admitting the experiment was headed in the wrong direction. Rolling back to earlier work and debugging everything myself was the moment things turned around. The takeaway is simple: LLMs are great collaborators and sounding boards, but they’re not oracles. Their advice is raw material, not gospel.

## **Continual Improvement**

If I continued developing this system, I already see several clear next steps.

**Technical improvements:**
I’d introduce a local frontier-class model, like GPT-OSS 20B, to fill a gap in the comparison. Right now, the choices are “fine-tune small models locally” or “pay for inference from big models.” Adding a self-hosted, high-capacity model would complete that picture.

I’d also revisit hyperparameters, especially for BART. My Frankenstein model got the bulk of my tuning energy, but BART likely has more performance in it.

**Data and feature enhancements:**
I’d explore structure-aware features — classifying dialogues by length, speaker count, or format, or at least giving models those hints. SAMSum is simple, but real-world chats vary more than I accounted for.

**Feature extensions and deployment considerations:**
If I were thinking about deployment, I’d experiment with user-level controls: adjustable summary length, tone, compression strength. And if the system were ever meant for broader use, I’d need better preprocessing for non-SAMSum formats and possibly multilingual handling.

Altogether, the project did what I wanted: it gave me a wide view of the summarization landscape, forced me into hands-on engineering challenges, and taught me a useful lesson about balancing intuition with machine-generated suggestions.
