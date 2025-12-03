"""
Builds the BERT encoder + GPT-2 decoder Frankenstein model.
"""

from transformers import EncoderDecoderModel


def build_bert_gpt2_model(
    gpt_pad_token_id: int,
    gpt_bos_token_id: int,
    decoder_tokenizer=None,
):
    """
    Creates a BERT encoder + GPT-2-medium decoder with cross-attention.

    We keep BERT at its native max_position_embeddings (=512) to avoid
    version-dependent hacks. Make sure MAX_SOURCE_LEN <= 512 in the notebook.
    """

    # 1. Create the Franken-model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "bert-base-uncased",
        "gpt2-medium",
    )

    # 2. Disable caching (better for training + ROCm)
    model.config.use_cache = False
    model.decoder.config.use_cache = False

    # 3. Core encoder-decoder config
    model.config.decoder_start_token_id = gpt_bos_token_id
    model.config.pad_token_id = gpt_pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # 4. If tokenizer has been modified (pad token added), resize GPT-2 embeddings
    if decoder_tokenizer is not None:
        model.decoder.resize_token_embeddings(len(decoder_tokenizer))

    # 5. Generation configuration
    gen_cfg = model.generation_config
    gen_cfg.pad_token_id = gpt_pad_token_id
    gen_cfg.bos_token_id = gpt_bos_token_id
    gen_cfg.max_length = 140
    gen_cfg.min_length = 5
    gen_cfg.no_repeat_ngram_size = 3
    gen_cfg.early_stopping = True
    gen_cfg.length_penalty = 2.0
    gen_cfg.num_beams = 4

    return model