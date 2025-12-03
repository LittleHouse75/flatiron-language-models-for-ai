"""
ROUGE evaluation utilities for summarization tasks.

Used by:
- Experiment 1 (optional)
- Experiment 2 (optional)
- Experiment 3 (required)
- Evaluation/Comparison notebook
"""

from typing import List, Dict
import pandas as pd

try:
    import evaluate
    _rouge = evaluate.load("rouge")
except Exception:
    _rouge = None


def compute_rouge_from_lists(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """
    Compute ROUGE scores given two lists of strings.
    
    Returns:
        dict with keys: rouge1, rouge2, rougeL, rougeLsum
    """
    if _rouge is None:
        raise ImportError(
            "HuggingFace 'evaluate' library not found. "
            "Install with: pip install evaluate"
        )
    
    results = _rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=use_stemmer,
    )
    return results


def compute_rouge_from_df(
    df: pd.DataFrame,
    pred_col: str = "model_summary",
    ref_col: str = "reference_summary",
) -> Dict[str, float]:
    """
    Convenience wrapper when predictions and references are in the same DataFrame.
    """
    preds = df[pred_col].tolist()
    refs = df[ref_col].tolist()
    return compute_rouge_from_lists(preds, refs)