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
    max_source_len,  # CHANGED: renamed for clarity
    max_target_len,
    source_prefix: str = "",  # NEW: optional prefix
):
    """
    Generate a summary for a single dialogue.
    
    Parameters
    ----------
    model : transformers model
        The encoder-decoder or seq2seq model
    encoder_tokenizer : tokenizer
        Tokenizer for encoding the input
    decoder_tokenizer : tokenizer
        Tokenizer for decoding the output
    text : str
        The dialogue text to summarize
    device : torch.device
        Device to run inference on
    max_source_len : int
        Maximum length for the source/input sequence
    max_target_len : int
        Maximum length for the generated summary
    source_prefix : str
        Optional prefix to prepend (e.g., "summarize: " for T5)
    """
    # Apply prefix if specified
    prefixed_text = source_prefix + text
    
    # Encode using the encoder tokenizer
    enc = encoder_tokenizer(
        prefixed_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_source_len,  # FIXED: use explicit length, not model_max_length
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_length=max_target_len,
        )

    # Decode using the decoder tokenizer
    return decoder_tokenizer.decode(out[0], skip_special_tokens=True)


def qualitative_samples(
    df,
    model,
    encoder_tokenizer,
    decoder_tokenizer,
    device,
    max_source_len,  # NEW: added this parameter
    max_target_len,
    source_prefix: str = "",  # NEW: optional prefix
    n=5,
):
    """
    Print n random qualitative examples comparing model output to human summaries.
    """
    print(f"--- {n} qualitative samples ---")

    samples = df.sample(n)
    for idx, row in samples.iterrows():
        dialog = row["dialogue"]
        ref = row["summary"]

        pred = generate_summary(
            model=model,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            text=dialog,
            device=device,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            source_prefix=source_prefix,
        )

        print(f"ID {idx}")
        print("DIALOGUE:", dialog[:300].replace("\n", " | "), "...")
        print("HUMAN:", ref)
        print("MODEL:", pred)
        print("-" * 80)
