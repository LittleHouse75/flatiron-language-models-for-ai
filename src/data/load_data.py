"""
Unified dataset loader for SAMSum.

Loads: train_df, val_df, test_df
Handles both parquet caching and HF downloads.

This file replaces the old load_samsum.py
and is used by all notebooks.
"""

import pandas as pd
from pathlib import Path
from datasets import load_dataset


def load_samsum(
    data_dir: str = "data/raw",
    use_local_first: bool = True,
    save_local: bool = True,
    force_refresh: bool = False,
):
    """
    Load SAMSum train/val/test as pandas DataFrames.

    Parameters
    ----------
    data_dir : str
        Directory where parquet versions of splits may live.
    use_local_first : bool
        If True, attempt to read existing parquet files before downloading.
    save_local : bool
        If True, save parquet versions after loading.
    force_refresh : bool
        If True, ignore local files and reload from HF.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "samsum_train.parquet"
    val_path = data_dir / "samsum_val.parquet"
    test_path = data_dir / "samsum_test.parquet"

    local_files_exist = (
        train_path.exists() and
        val_path.exists() and
        test_path.exists()
    )

    # Try local first
    if use_local_first and local_files_exist and not force_refresh:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
        return train_df, val_df, test_df

    # Otherwise load from HF
    ds = load_dataset("knkarthick/samsum")
    train_df = ds["train"].to_pandas()
    val_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()

    # Save to parquet for faster reuse
    if save_local:
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

    return train_df, val_df, test_df