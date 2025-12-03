"""
Builder for T5 summarization model.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer


def build_t5_model(model_name="t5-small"):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer