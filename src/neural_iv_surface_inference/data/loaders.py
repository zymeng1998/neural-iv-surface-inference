"""Dataset loading utilities for IV surface benchmark data.

Provides a PyTorch Dataset that wraps benchmark Parquet files from Step 4.
Each sample is a single surface point with coordinate features, target IV,
observation mask, and split label.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Only load columns needed for training / evaluation
_REQUIRED_COLUMNS = [
    "date", "tau", "log_moneyness", "implied_volatility",
    "iv_clean", "observed", "split", "noise_sigma",
]


class IVSurfaceDataset(Dataset):
    """PyTorch Dataset for IV surface point data.

    Each sample is a single observation point with:
        features : (log_moneyness, tau)  — float32, shape (2,)
        target   : iv_clean              — float32, scalar
        observed : bool                  — whether this point is "observed"
        iv_input : implied_volatility    — noisy IV (for observed) or clean (for unobserved)

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataset (output of Step 4) filtered to the desired split.
    """

    def __init__(self, df: pd.DataFrame):
        self.log_moneyness = torch.tensor(
            df["log_moneyness"].values, dtype=torch.float32
        )
        self.tau = torch.tensor(df["tau"].values, dtype=torch.float32)
        self.iv_clean = torch.tensor(
            df["iv_clean"].values, dtype=torch.float32
        )
        self.iv_input = torch.tensor(
            df["implied_volatility"].values, dtype=torch.float32
        )
        self.observed = torch.tensor(df["observed"].values, dtype=torch.bool)

        # Store date indices for per-date evaluation
        dates = df["date"].values
        unique_dates = np.unique(dates)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        self.date_idx = torch.tensor(
            [date_to_idx[d] for d in dates], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.iv_clean)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.stack([
                self.log_moneyness[idx], self.tau[idx]
            ]),
            "target": self.iv_clean[idx],
            "iv_input": self.iv_input[idx],
            "observed": self.observed[idx],
            "date_idx": self.date_idx[idx],
        }


def load_benchmark_splits(
    path: Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader | pd.DataFrame]:
    """Load a benchmark Parquet and return DataLoaders + raw DataFrames.

    Parameters
    ----------
    path : Path
        Path to a benchmark Parquet file (from Step 4).
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    dict with keys:
        'train_loader', 'val_loader', 'test_loader' — DataLoader instances
        'train_df', 'val_df', 'test_df' — raw DataFrames (for interpolation baseline)
    """
    # Read only needed columns to reduce memory
    import pyarrow.parquet as pq

    available_cols = pq.read_schema(path).names
    cols_to_read = [c for c in _REQUIRED_COLUMNS if c in available_cols]
    df = pd.read_parquet(path, columns=cols_to_read)

    result = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].reset_index(drop=True)
        result[f"{split}_df"] = split_df

        ds = IVSurfaceDataset(split_df)
        # Only shuffle training data
        result[f"{split}_loader"] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Free the full dataframe — splits are now held independently
    del df
    gc.collect()

    return result
