"""
ROUGE evaluation utilities for summarization tasks.

This is the SINGLE SOURCE OF TRUTH for ROUGE computation.
All notebooks and training code should use these functions
to ensure consistent evaluation across experiments.
"""

from typing import List, Dict
import pandas as pd

try:
    import evaluate
    _rouge = evaluate.load("rouge")
except Exception:
    _rouge = None

# =============================================================================
# CONFIGURATION: These settings are used everywhere for consistency
# =============================================================================
DEFAULT_USE_STEMMER = True


def compute_rouge_from_lists(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = DEFAULT_USE_STEMMER,
) -> Dict[str, float]:
    """
    Compute ROUGE scores given two lists of strings.
    
    This is the core function - all other ROUGE functions call this one.
    
    Parameters
    ----------
    predictions : List[str]
        Model-generated summaries
    references : List[str]
        Human-written reference summaries
    use_stemmer : bool
        Whether to use Porter stemmer for word matching.
        Default is True for consistency across all experiments.
    
    Returns
    -------
    dict
        Dictionary with keys: rouge1, rouge2, rougeL, rougeLsum
        Values are floats between 0 and 1.
    
    Example
    -------
    >>> preds = ["The cat sat on the mat.", "Dogs are great pets."]
    >>> refs = ["A cat was sitting on a mat.", "Dogs make wonderful pets."]
    >>> scores = compute_rouge_from_lists(preds, refs)
    >>> print(f"ROUGE-L: {scores['rougeL']:.3f}")
    """
    if _rouge is None:
        raise ImportError(
            "HuggingFace 'evaluate' library not found. "
            "Install with: pip install evaluate"
        )
    
    # Validate inputs
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(references)} references"
        )
    
    if len(predictions) == 0:
        raise ValueError("Cannot compute ROUGE on empty lists")
    
    # This prevents cryptic errors from malformed data
    def clean_text(text):
        if text is None:
            return ""
        if isinstance(text, float) and pd.isna(text):
            return ""
        return str(text)
    
    predictions = [clean_text(p) for p in predictions]
    references = [clean_text(r) for r in references]
    
    results = _rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=use_stemmer,
    )
    
    return results


def compute_rouge_from_df(
    df,  # pd.DataFrame - not type hinted to avoid pandas import at module level
    pred_col: str = "model_summary",
    ref_col: str = "reference_summary",
    use_stemmer: bool = DEFAULT_USE_STEMMER,
) -> Dict[str, float]:
    """
    Convenience wrapper when predictions and references are in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing prediction and reference columns
    pred_col : str
        Name of column containing model predictions
    ref_col : str
        Name of column containing reference summaries
    use_stemmer : bool
        Whether to use Porter stemmer (default: True)
    
    Returns
    -------
    dict
        ROUGE scores dictionary
    """
    # Validate columns exist
    if pred_col not in df.columns:
        raise KeyError(f"Prediction column '{pred_col}' not found in DataFrame")
    if ref_col not in df.columns:
        raise KeyError(f"Reference column '{ref_col}' not found in DataFrame")
    
    preds = df[pred_col].tolist()
    refs = df[ref_col].tolist()
    
    return compute_rouge_from_lists(preds, refs, use_stemmer=use_stemmer)
