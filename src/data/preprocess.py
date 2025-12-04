"""
Tokenization + dataset classes shared across models.
"""

import torch
from torch.utils.data import Dataset


class SummaryDataset(Dataset):
    """
    General-purpose sequence-to-sequence dataset.

    Supports:
    - Any encoder tokenizer
    - Any decoder tokenizer
    - Modern HF 5.x `text_target=` API
    - Automatic masking of padding tokens for labels
    - Optional source prefix (e.g., "summarize: " for T5)
    - Returns decoder_attention_mask for models that need it  # <-- NEW
    """

    def __init__(
        self,
        df,
        encoder_tokenizer,
        decoder_tokenizer,
        max_source_len: int,
        max_target_len: int,
        source_prefix: str = "",
    ):
        self.df = df.reset_index(drop=True)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source_prefix = source_prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Apply prefix if specified
        src_text = self.source_prefix + str(row["dialogue"])
        tgt_text = str(row["summary"])

        # =================================================================
        # FIX: Explicitly set padding side before tokenizing
        # This ensures consistency regardless of tokenizer defaults
        # =================================================================
        
        # Store original padding sides to restore after
        original_enc_padding = self.encoder_tokenizer.padding_side
        original_dec_padding = self.decoder_tokenizer.padding_side
        
        try:
            # Use RIGHT padding for encoder-decoder models during training
            # This puts real tokens at the start, padding at the end
            self.encoder_tokenizer.padding_side = "right"
            self.decoder_tokenizer.padding_side = "right"
            
            # Encoder side: BERT or BART or T5 encoder
            enc = self.encoder_tokenizer(
                src_text,
                max_length=self.max_source_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Decoder side: GPT-2, BART, or T5 decoder
            dec = self.decoder_tokenizer(
                text_target=tgt_text,
                max_length=self.max_target_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        finally:
            # ALWAYS restore original settings
            # This prevents side effects if tokenizer is used elsewhere
            self.encoder_tokenizer.padding_side = original_enc_padding
            self.decoder_tokenizer.padding_side = original_dec_padding

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Retrieve labels and the decoder mask
        labels = dec["input_ids"].squeeze(0).clone()
        decoder_attention_mask = dec["attention_mask"].squeeze(0)

        # Mask padding positions for loss calculation
        labels[decoder_attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
