"""
Builder for BART summarization model.
"""

from transformers import BartForConditionalGeneration, BartTokenizer


def build_bart_model(model_name="facebook/bart-base"):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer