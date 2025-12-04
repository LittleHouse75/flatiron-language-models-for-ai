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
    """

    def __init__(
        self,
        df,
        encoder_tokenizer,
        decoder_tokenizer,
        max_source_len: int,
        max_target_len: int,
        source_prefix: str = "",  # NEW: optional prefix for T5-style models
    ):
        self.df = df.reset_index(drop=True)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source_prefix = source_prefix  # NEW

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # NEW: Apply prefix if specified
        src_text = self.source_prefix + str(row["dialogue"])
        tgt_text = str(row["summary"])

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

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Retrieve labels and the decoder mask
        labels = dec["input_ids"].squeeze(0).clone()
        dec_mask = dec["attention_mask"].squeeze(0)

        # Mask padding positions for loss calculation.
        # We use attention_mask (not token IDs) because pad_token may equal eos_token.
        # attention_mask=1 means "real token" (including the true EOS), 
        # attention_mask=0 means "padding" (should be ignored in loss).
        labels[dec_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
