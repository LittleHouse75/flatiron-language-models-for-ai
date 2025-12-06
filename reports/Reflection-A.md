Tim Nevits
Flatiron School
AI / ML Program

# Capstone 3 Self Reflection (300-500 Words)

## **Accomplishments**

This project ended up being bigger and more ambitious than I expected, and I’m genuinely satisfied with how much of it I pushed across the finish line. On the technical side, the thing I’m most proud of is getting the Frankenstein DistilBERT → DistilGPT-2 encoder–decoder model into genuinely “viable baseline” territory. That architecture is held together with wires and goodwill, and the cross-attention layers start out as random soup, so making it competitive felt like a real win. The turning point was discovering that my repeated-n-gram penalty was quietly killing the model’s ability to generate clean summaries. Removing it bumped ROUGE scores by around 10 points, and everything clicked from there.

I’m also pleased with the scope. I didn’t just train one model; I trained three, then compared them against five frontier API models. It gave me a complete view of the tradeoffs between local fine-tuning and off-the-shelf LLMs. I also feel like I kept the project coherent from start to finish — EDA → experiments → evaluation → comparison — without losing the thread along the way.

## **Opportunity for Growth**

The biggest challenge, and the clearest lesson learned, came from over-trusting LLM suggestions early on. I let Claude push several training and architectural decisions that ultimately sent performance in the wrong direction. That cost me about two days before I rolled back to an older checkpoint and reclaimed control. The upside is that it forced me to recalibrate my relationship with LLMs: they’re great collaborators, but they’re not project leads. I need to validate aggressively and trust my own understanding more than I did in those early stages.

## **Feedback Request**

What I’d really like to know is whether the narrative holds together. The project starts with the idea of attention overload, moves into three solution families, and then lands on a tradeoff analysis. Does that structure feel coherent and motivated, or does it feel like too much ground for one project? I’m also curious if you see any angles I missed — either experiments I should have included or ways to tighten the framing — especially since the comparison spans eight different models. Any outside perspective on the clarity and pacing of the story would be helpful.