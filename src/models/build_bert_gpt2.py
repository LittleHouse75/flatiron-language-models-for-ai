"""
Builds the BERT encoder + GPT-2 (small, 124M) decoder Frankenstein model.
"""

from transformers import EncoderDecoderModel


def build_bert_gpt2_model(
    gpt_pad_token_id: int,
    gpt_bos_token_id: int,
    decoder_tokenizer=None,
    max_length: int = 128,  # NEW: configurable parameter
):
    """
    Build a BERT encoder + GPT-2 decoder model.
    
    Parameters
    ----------
    gpt_pad_token_id : int
        Token ID for padding (usually same as eos_token_id for GPT-2)
    gpt_bos_token_id : int
        Token ID for beginning of sequence
    decoder_tokenizer : tokenizer, optional
        If provided, resize embeddings to match tokenizer vocabulary
    max_length : int
        Maximum generation length (should match your MAX_TARGET_LEN)
    """
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "bert-base-uncased",
        "gpt2",
    )

    # Disable caching (needed for training/gradient checkpointing)
    model.config.use_cache = False
    model.decoder.config.use_cache = False

    # Core encoder-decoder config
    model.config.decoder_start_token_id = gpt_bos_token_id
    model.config.pad_token_id = gpt_pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Resize embedding layer if tokenizer was expanded (pad token)
    if decoder_tokenizer is not None:
        model.decoder.resize_token_embeddings(len(decoder_tokenizer))

    # Generation defaults - use passed max_length
    gen_cfg = model.generation_config
    gen_cfg.pad_token_id = gpt_pad_token_id
    gen_cfg.bos_token_id = gpt_bos_token_id
    gen_cfg.max_length = max_length  # CHANGED: use parameter
    gen_cfg.min_length = 5
    gen_cfg.no_repeat_ngram_size = 3
    gen_cfg.early_stopping = True
    gen_cfg.length_penalty = 2.0
    gen_cfg.num_beams = 4

    return model
