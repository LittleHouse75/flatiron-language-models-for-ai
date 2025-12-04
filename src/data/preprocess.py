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

        # Retrieve labels and the decoder mask
        labels = dec["input_ids"].squeeze(0).clone()
        dec_mask = dec["attention_mask"].squeeze(0)

        # CRITICAL FIX:
        # Instead of looking for the specific pad_token_id (which might be the same as EOS),
        # we use the attention_mask to find which tokens are padding (0).
        # 1 = real token (including the first EOS), 0 = padding
        labels[dec_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
