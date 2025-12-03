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
    """

    def __init__(
        self,
        df,
        encoder_tokenizer,
        decoder_tokenizer,
        max_source_len: int,
        max_target_len: int,
    ):
        self.df = df.reset_index(drop=True)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        src_text = str(row["dialogue"])
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

        labels = dec["input_ids"].squeeze(0)
        # Mask PAD tokens for cross-entropy loss
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }