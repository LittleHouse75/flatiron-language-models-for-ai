
Tim Nevits
AI / ML Engineer in Training
Flatiron School
November 29, 2025


# Project Pitch: Dialogue Summarization with Transformer Models

## Problem Statement and Proposed Solution

### Problem Description

In modern workplaces, chat tools were supposed to replace chaotic email threads with something more focused and “real-time.” In practice, they often stacked a new layer of chaos on top of the old one.

When I worked at Visa, we used Slack and later Microsoft Teams. My day was frequently dominated not by actual work, but by trying to reconstruct what had already happened in long group chats. People would add me to a channel midstream with an implicit expectation: _“You’re in this now, so you’re caught up.”_ In reality, that meant scrolling through pages of back-and-forth, half-threads, and side jokes just to figure out:

- What decision was actually made?
- Who owns the next step?
- Is there something I’m supposed to do here?

That pattern wasn’t unique to me. Research consistently shows that information overload is now the norm rather than the exception. A Gartner-backed survey found that **38% of employees say they receive an “excessive” volume of communications** at their organization.  [oai_citation:0‡Harvard Business Review](https://hbr.org/2023/05/reducing-information-overload-in-your-organization?utm_source=chatgpt.com) Microsoft’s telemetry data shows that employees using Microsoft 365 are **interrupted roughly every two minutes by meetings, emails, or chat notifications**, adding up to about **275 interruptions per day**.  [oai_citation:1‡HR Dive](https://www.hrdive.com/news/workers-battle-an-infinite-workday-microsoft/751432/?utm_source=chatgpt.com) And a recent communication study estimates that employees spend **hours every day** dealing with digital communication alone, much of which they perceive as low-value.  [oai_citation:2‡EmailTooltester.com](https://www.emailtooltester.com/en/blog/work-communications-burnout/?utm_source=chatgpt.com)

So the subjective feeling of “I spend my day hunting through chats instead of doing my job” is backed by data: high message volume, fragmented attention, and a lot of time spent decoding text rather than acting on it.

### Impact Assessment

The impact of this overload shows up in three main ways.

**User experience and satisfaction.**  
When you open a chat and are greeted by a wall of text, your first emotion usually isn’t “how productive.” It’s closer to dread. You know you’ll need to scroll back, try to infer context, and manually assemble an internal summary. That costs mental energy before you’ve even started your real work. Over time, that kind of constant cognitive tax contributes to frustration, stress, and burnout. Digital communication has been directly linked to increased fatigue and stress in multiple workplace surveys.  [oai_citation:3‡action.deloitte.com](https://action.deloitte.com/insight/2483/reliance-on-digital-communication-is-stressing-the-workforce?utm_source=chatgpt.com)

**Retention and platform engagement.**  
If people associate the messaging platform with confusion and overwhelm, they disengage in subtle ways. They skim instead of reading, miss details, or avoid channels that feel “noisy.” Studies on communication quality show that poor or excessive communication is a major contributor to missed deadlines, morale problems, and turnover.  [oai_citation:4‡Simon & Simon International](https://www.simonandsimon.co.uk/blog/workplace-communication-statistics?utm_source=chatgpt.com) In the worst cases, employees decide that the environment itself is unsustainable, not because the _work_ is impossible, but because staying “caught up” requires too much overhead.

**Competitive disadvantage.**  
Companies that can surface the right information at the right time will move faster and make fewer mistakes. Those that drown their people in undifferentiated noise will lose time to clarification, rework, and misunderstandings. Harvard Business Review, summarizing Gartner research, notes that information overload is already a recognized organizational risk: high volumes of communication actively undermine productivity if not managed.  [oai_citation:5‡Harvard Business Review](https://hbr.org/2023/05/reducing-information-overload-in-your-organization?utm_source=chatgpt.com) A messaging platform that can automatically summarize conversations gains a clear edge over one that simply delivers more messages.

### Solution Vision

The core idea is simple: instead of forcing humans to summarize chats in their heads every time they open a channel, **let the system do the first pass**.

An automated dialogue summarization feature would:

- Generate a concise 1–3 sentence summary for each conversation or time window.
- Emphasize the main topic, decisions, and action items.
- Update when new messages arrive, so users always have an at-a-glance “state of the thread.”
- Allow users to drill down into details when needed, instead of starting from raw logs.

In other words, we flip the typical LLM workflow. Instead of users carefully crafting context for the model, the model continuously crafts context for the user.

The benefits line up directly with the business problem:

- **Reduced cognitive load:** Users see the “headline and bullet points” before they see the entire scrollback.
- **More accessible conversations:** New joiners or returning participants can re-enter a thread without manually reconstructing history.
- **Stronger value proposition:** The platform stops being a firehose and starts acting like an intelligent assistant.
- **Premium features:** Advanced summarization (e.g., department-specific policies, adjustable detail level, or meeting-ready summaries) could be layered as paid features.

### Success Criteria

To keep this grounded, I’ll define success along three dimensions: quality, user experience, and performance.

- **Quality (ROUGE metrics).**  
  On the SAMSum test set, target:
  - **ROUGE-1 ≥ 40**
  - **ROUGE-L ≥ 35**  
  These won’t match state-of-the-art research numbers, but they are reasonable goals for a capstone-level prototype and sufficient to demonstrate feasibility.

- **User experience.**  
  - Summaries should almost always capture the main topic and any explicit decisions or action items.
  - Summaries should be short enough to read in a few seconds (roughly 1–3 sentences).

- **Technical performance.**  
  - On a single consumer GPU, batch inference latency should be well under **2000 ms per conversation** for the smaller models.
  - API-based models should still feel interactive when used on-demand (sub-second to a few seconds per summary).
  - The system should be structured so that frequent, high-traffic channels can have summaries precomputed or cached.

---

## Problem-Solving Process

### Process Framework

To explore this problem end-to-end, I’ll follow a 7-step process:

1. **Data Exploration and Preparation**  
   Load the SAMSum dataset, inspect dialogue and summary distributions, analyze length, number of speakers, and structure. Build a preprocessing pipeline that handles tokenization, truncation, and train/validation/test splits.

2. **Model Architecture Design (Three Experiments)**  
   Implement three contrasting approaches:
   - **Experiment 1:** Custom encoder–decoder with a BERT encoder and GPT-2 decoder (meeting the assignment’s “BERT + autoregressive” requirement, even though they weren’t pretrained together).
   - **Experiment 2:** Purpose-built sequence-to-sequence models like BART and T5 that already have pretrained encoder–decoder attention.
   - **Experiment 3:** Frontier autoregressive models (e.g., GPT 5.1, Gemini 3 Pro, Claude Opus 4.5) accessed via API with different prompting strategies.

3. **Training and Optimization**  
   For Experiments 1 and 2, fine-tune on SAMSum with consistent hyperparameters where reasonable: learning rate, batch size, gradient accumulation, and number of epochs. Use validation ROUGE to guide early stopping.

4. **Evaluation and Testing**  
   Use ROUGE-1, ROUGE-2, and ROUGE-L as primary metrics. Evaluate on the held-out test set and collect qualitative examples. For API models, evaluate on the same test split to keep comparisons fair.

5. **Error and Behavior Analysis**  
   Manually review a sample of outputs for each model, categorize errors (missing key info, hallucinations, wrong speaker attribution, etc.), and analyze performance by dialogue complexity (length, number of turns, number of speakers).

6. **Deployment Considerations and Trade-Offs**  
   Think through how each approach would behave in a real product: latency, scalability, cost, privacy, and maintainability. Compare fine-tuned local models to API-based solutions from an engineering standpoint.

7. **Documentation and Communication**  
   Consolidate findings into a Jupyter notebook, a written report, and a short video presentation. Emphasize not just which model “won,” but how the trade-offs map back to business needs.

### Conceptual Representation

At a high level, all three experiments share the same outer pipeline:

```text
[Raw Dialogue Text]
          |
          v
[Preprocessing: tokenization, truncation, attention masks]
          |
          v
[Model (varies by experiment)]
          |
          v
[Generated Summary Text]
          |
          v
[Evaluation: ROUGE + qualitative review]
```

The interesting differences are inside that “Model” box. Each experiment gets its own mini-flowchart.

Experiment 1: Custom Encoder–Decoder (BERT → GPT-2)

```text
[Dialogue Text]
        |
        v
[BERT Tokenizer]
        |
        v
[BERT Encoder (pretrained)]
        |
        v
[Cross-Attention Layers (random init)]
        |
        v
[GPT-2 Decoder (pretrained autoregressive)]
        |
        v
[Generated Summary]
```

Here, the encoder and decoder both start from strong pretrained checkpoints, but the cross-attention that lets them talk to each other is randomly initialized, which is a known training bottleneck.

Experiment 2: Purpose-Built Seq2Seq (BART / T5)

```text
[Dialogue Text]
        |
        v
[BART/T5 Tokenizer]
        |
        v
[Seq2Seq Encoder (pretrained)]
        |
        v
[Seq2Seq Decoder + Cross-Attention (pretrained)]
        |
        v
[Generated Summary]
```

In this case, the encoder and decoder were trained together from the start for sequence-to-sequence tasks (like denoising, translation, and summarization), so the cross-attention is already well-behaved.

Experiment 3: Frontier LLM APIs (GPT-4o, Claude 3.5, etc.)

```text
[Dialogue Text]
        |
        v
[Prompt Construction
 (zero-shot / instructions / few-shot)]
        |
        v
[Frontier LLM via API (autoregressive)]
        |
        v
[Generated Summary]
        |
        v
[Evaluation + Cost / Latency Tracking]
```

Here, there is no training in the traditional sense. The main levers are prompt design, temperature, and model choice.

---

### **Methodology Justification**

The assignment specifically calls for “an encoder-decoder architecture using pre-trained BERT models” and “auto-regressive modeling with ChatGPT.” The three-experiment design is my way of honoring that requirement while also exploring the real design space.

* **Why BERT → GPT-2 in Experiment 1?**
  BERT is a natural choice for an encoder: it’s bidirectional and good at digesting input text. GPT-2 is a natural choice for an autoregressive decoder: it’s designed to generate fluent text one token at a time. Hugging Face’s EncoderDecoderModel makes it possible to glue them together. That configuration directly satisfies the “BERT + autoregressive” requirement while also exposing the practical challenge that the cross-attention layers start off untrained.

* **Why BART / T5 in Experiment 2?**
  BART and T5 represent what people actually use in production for summarization. They are pretrained as full encoder–decoder models, with the encoder and decoder learning to cooperate through cross-attention from the beginning. They give me a strong baseline and let me answer the question: *“How much better is it to start from a purpose-built seq2seq model instead of a glued-together pair?”*

* **Why frontier LLMs in Experiment 3?**
  The “auto-regressive modeling with ChatGPT” part of the assignment is naturally satisfied by using ChatGPT-style models as summarizers via API. This reflects modern industry practice: before investing in custom fine-tuning, many teams start with prompt-engineered calls to GPT-4-class models. Comparing their summaries (and their cost/latency) to fine-tuned local models gives a more realistic business picture.

* **Why fine-tuning instead of training from scratch?**
  Training transformer models from scratch on SAMSum alone would be both computationally expensive and technically unnecessary. Fine-tuning lets me leverage large-scale pretraining and adapt it to dialogue summarization using a relatively modest setup (single consumer GPU).

* **Why ROUGE and qualitative review?**
  ROUGE is still the standard automatic metric in summarization research and aligns with the project rubric. At the same time, ROUGE alone can’t catch hallucinations, speaker confusions, or subtle factual errors, so I’ll complement it with manual error categorization and qualitative examples.

---

### **Alignment with Requirements**

The project’s three-experiment design is deliberately mapped to the assignment’s requirements:

* **“Implement an encoder-decoder architecture using pre-trained BERT and auto-regressive with ChatGPT models.”**
  * Met by **Experiment 1**, which uses a BERT-based encoder with an autoregressive GPT-2 decoder.
  * Reinforced by **Experiment 3**, which uses ChatGPT-style autoregressive models via API.

* **“Create a complete pipeline: data loading, preprocessing, model training, evaluation, and inference.”**
  * All three experiments share a common preprocessing and evaluation pipeline using SAMSum and ROUGE.
  * Experiments 1 and 2 involve full training loops (with early stopping and checkpointing).
  * Experiment 3 uses inference-only pipelines and focuses on prompt engineering.

* **“Evaluate summarization quality and interpret model performance.”**
  * I will report ROUGE scores across all three experiments.
  * I’ll compare models on error types, length calibration, hallucination heuristics, and performance on more complex dialogues.

* **“Connect technical results back to business value.”**
  * Each model family will be evaluated not just on ROUGE, but also on cost, latency, scalability, and privacy—exactly the dimensions that matter for a messaging platform like “Acme Communications.”

* **Deliverables: pitch, MVP notebook, final notebook, report, video.**
  * This pitch defines the plan and business framing.
  * The MVP and final submissions will consist of a structured notebook (or set of notebooks), a README-style report, and a short video walkthrough of the results.

In short, the plan follows the letter of the assignment while also expanding it into a more realistic comparison study.

---

## **Timeline and Scope**

Today is **Saturday, November 29, 2025**, and some groundwork is already in place (environment setup, ROCm debugging, early EDA).

### **Research and Preparation Phase (Completed / In Progress)**

**Tuesday, Nov 25 – Saturday, Nov 29, 2025** 
* Read the SAMSum paper and dataset documentation.
* Review Hugging Face docs on encoder–decoder models, BART, T5, and summarization pipelines.
* Explore evaluation metrics (ROUGE) and set up the evaluate library.
* Run initial EDA on SAMSum:
  * Dialogue length distributions
  * Summary lengths
  * Number of speakers and turns

This phase is effectively done; the remaining work is implementation and analysis.

### **Implementation Phases**

**Experiment 1: BERT → GPT-2 Custom Encoder–Decoder** 
* **Sunday, Nov 30, 2025**
  * Finalize preprocessing functions for input/target sequences.
  * Implement EncoderDecoderModel with BERT encoder and GPT-2 decoder.
  * Run at least one full training run on a subset, then scale up as compute allows.
  * Save best-performing checkpoint and log ROUGE on validation.

**Experiment 2: BART / T5 Seq2Seq Models** 
* **Monday, Dec 1, 2025**
  * Implement training loops for facebook/bart-base and t5-small (or flan-t5-small).
  * Reuse the same preprocessing pipeline where possible.
  * Train for fewer epochs (these models typically converge faster).
  * Save metrics, checkpoints, and validation ROUGE.

**Experiment 3: Frontier LLM APIs** 
* **Tuesday, Dec 2, 2025**
  * Implement zero-shot, instructed, and few-shot prompting schemes.
  * Sample ~800 test dialogues (or the full SAMSum test split) and log:
    * Summaries
    * ROUGE scores
    * Latency per call
    * Approximate cost per 1,000 summaries based on token estimates.

**Evaluation and Analysis** 
* **Wednesday, Dec 3, 2025**
  * Consolidate results into a comparison table.
  * Run additional analyses:
    * Error categorization on a sample of outputs per model.
    * Length calibration plots (predicted vs. reference summary lengths).
    * Simple hallucination heuristics using named entities.
    * Performance by dialogue complexity (length, turns, speakers).

**Documentation, Report, and Video** 
* **Thursday, Dec 4, 2025**
  * Clean and re-run the final notebook(s) top-to-bottom.
  * Write the project report (README or separate PDF) summarizing:
    * Business context
    * Methodology
    * Results
    * Limitations
    * Future work
  * Record a 7–10 minute video walkthrough and upload.
  * Submit project materials.

### **Iteration Points**

If any major issues appear (e.g., BERT→GPT-2 training underperforms badly, or BART/T5 runs out of memory), I’ll:
* Simplify or downsize models (e.g., smaller variants).
* Prioritize making at least one seq2seq baseline solid, then treat others as ablations.
* Focus extra time on clearer analysis rather than squeezing in more half-baked experiments.

There is some built-in slack:
* **Friday, Dec 5 – Saturday, Dec 6, 2025** can serve as contingency for reruns, bug fixes, or minor improvements before any absolute course deadlines.

### **Risk Management**

**Compute resource limitations.** 

All training happens on a single consumer AMD GPU (ROCm stack). That means I can’t run multiple large experiments in parallel, and I need to be careful with batch sizes and model choices. The mitigation is to use smaller model variants (e.g., t5-small, bart-base), gradient accumulation, and aggressive monitoring of GPU memory.

**Architecture difficulties in Experiment 1.** 

The BERT → GPT-2 configuration is known to train more slowly because the cross-attention is randomly initialized. It might underperform relative to BART/T5. That’s acceptable as long as I document the behavior and show that this is an architectural limitation, not just a tuning issue.

**API limits and costs.** 

For Experiment 3, there are rate limits and usage costs. I’ll:

* Use a reasonable sample size (e.g., the official test split) rather than the full dataset for heavy models.
* Cache results locally to avoid repeated calls.
* Estimate cost per 1,000 summaries to demonstrate feasibility, rather than trying to process huge volumes.

**Privacy and deployment concerns.** 

In a real organization, sending raw chat logs to third-party APIs raises privacy and compliance questions—especially in finance or healthcare. A production system would need:
* Strong data-handling agreements if using external APIs.
* Options to run models fully on-prem (e.g., fine-tuned BART/T5).
* Clear settings for which channels are summarized (e.g., opt-in for private DMs).

**Timeline risk.** 

The plan assumes each major experiment can be run in roughly a day. If that slips, the priority will be:

1. At least one working custom encoder–decoder (Experiment 1).
2. At least one strong seq2seq baseline (BART or T5).
3. At least one frontier model configuration (e.g., GPT-4o-mini with instruction-style prompting).
⠀
The analysis and report will highlight quality vs. cost vs. control, even if not every possible model variant is explored.

### **Final Delivery**

* **Project critique / MVP-style notebook:** effectively covered by the first full working version of Experiments 1 and 2 (target: early next week).
* **Final implementation completion:** core experiments and analysis by **Wednesday, Dec 3, 2025**.
* **Documentation and video preparation:** **Thursday, Dec 4, 2025**.
* **Final submission:** by end of day **Thursday, Dec 4, 2025**, with **Friday–Saturday (Dec 5–6, 2025)** reserved as contingency if any minor fixes or re-uploads are allowed.

This plan gives me enough structure to satisfy the formal requirements while leaving room for the messy reality of training models on a single GPU at home.