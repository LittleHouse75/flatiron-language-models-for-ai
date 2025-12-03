"""
Qualitative inspection utilities.
"""

import torch

def generate_summary(
    model,
    encoder_tokenizer,
    decoder_tokenizer,
    text,
    device,
    max_target_len,
):
    # Encode using the *encoder* tokenizer (BERT)
    enc = encoder_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=encoder_tokenizer.model_max_length,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            generation_config=model.generation_config,
            max_length=max_target_len,
        )

    # Decode using the *decoder* tokenizer (GPT-2)
    return decoder_tokenizer.decode(out[0], skip_special_tokens=True)


def qualitative_samples(
    df,
    model,
    encoder_tokenizer,
    decoder_tokenizer,
    device,
    max_target_len,
    n=5,
):
    print(f"--- {n} qualitative samples ---")

    samples = df.sample(n)
    for idx, row in samples.iterrows():
        dialog = row["dialogue"]
        ref = row["summary"]

        pred = generate_summary(
            model,
            encoder_tokenizer,
            decoder_tokenizer,
            dialog,
            device,
            max_target_len,
        )

        print(f"ID {idx}")
        print("DIALOGUE:", dialog[:300].replace("\n", " | "), "...")
        print("HUMAN:", ref)
        print("MODEL:", pred)
        print("-" * 80)