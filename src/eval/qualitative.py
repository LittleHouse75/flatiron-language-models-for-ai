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
    max_source_len, 
    max_target_len,
    source_prefix: str = "",
):
    """
    Generate a summary for a single dialogue.
    
    [... existing docstring ...]
    """
    # Apply prefix if specified
    prefixed_text = source_prefix + text
    
    # Store original settings to restore later
    original_enc_padding = encoder_tokenizer.padding_side
    original_dec_padding = decoder_tokenizer.padding_side
    
    try:
        encoder_tokenizer.padding_side = "right"
        decoder_tokenizer.padding_side = "right"
        
        # Encode the input
        enc = encoder_tokenizer(
            prefixed_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_source_len,
        ).to(device)

        # Generate with no_grad for inference efficiency
        model.eval()
        with torch.no_grad():
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_length=max_target_len,
            )

        # Decode the output (move to CPU first to free GPU memory)
        out_cpu = out.cpu()
        summary = decoder_tokenizer.decode(out_cpu[0], skip_special_tokens=True)
        
        # Explicitly delete tensors to free GPU memory
        del enc, out
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return summary
    
    finally:
        # ALWAYS restore original settings
        encoder_tokenizer.padding_side = original_enc_padding
        decoder_tokenizer.padding_side = original_dec_padding


def qualitative_samples(
    df,
    model,
    encoder_tokenizer,
    decoder_tokenizer,
    device,
    max_source_len,
    max_target_len,
    source_prefix: str = "",
    n=5,
    seed=42,  # Add seed parameter with default value
):
    """
    Print n random qualitative examples comparing model output to human summaries.
    
    Parameters
    ----------
    seed : int or None
        Random seed for reproducible sampling. Set to None for different samples each run.
    """
    print(f"--- {n} qualitative samples (seed={seed}) ---")

    # Use the seed for reproducible sampling
    samples = df.sample(n, random_state=seed)
    
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
