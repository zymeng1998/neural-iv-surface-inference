"""Time-based train / validation / test splitting for IV surface data.

Splits are strictly chronological to prevent look-ahead bias:
  - Train  : earliest dates  (default 70%)
  - Val    : middle dates    (default 15%)
  - Test   : latest dates    (default 15%)

Also provides benchmark dataset versioning: named combinations of
(masking strategy, keep_frac, noise regime) saved as self-describing
Parquet files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time-based splitting
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add a ``split`` column ('train' / 'val' / 'test') based on date order.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *date_col* as datetime-like.
    train_frac : float
        Fraction of *unique dates* for training.
    val_frac : float
        Fraction for validation.  Remainder goes to test.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame with new ``split`` column.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    dates = df[date_col].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Map each date to its split via a lookup Series
    split_labels = pd.Series("test", index=dates)
    split_labels.iloc[:n_train] = "train"
    split_labels.iloc[n_train : n_train + n_val] = "val"

    df["split"] = df[date_col].map(split_labels).values
    return df


def split_summary(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a summary table of split statistics.

    Columns: split, n_dates, n_rows, date_min, date_max
    """
    rows = []
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        if len(sub) == 0:
            continue
        rows.append({
            "split": split_name,
            "n_dates": sub[date_col].nunique(),
            "n_rows": len(sub),
            "date_min": sub[date_col].min(),
            "date_max": sub[date_col].max(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark dataset versioning
# ---------------------------------------------------------------------------

def benchmark_name(
    strategy: str,
    keep_frac: float,
    noise_regime: str,
    heteroscedastic: bool = False,
) -> str:
    """Generate a canonical name for a benchmark variant.

    Examples
    --------
    >>> benchmark_name('random', 0.2, 'none')
    'spy_phase1_random20_noise0'
    >>> benchmark_name('realistic', 0.4, 'medium', heteroscedastic=True)
    'spy_phase1_realistic40_noisemed_hetero'
    """
    keep_pct = int(round(keep_frac * 100))

    noise_short = {
        "none": "0",
        "low": "low",
        "medium": "med",
        "high": "high",
    }.get(noise_regime, noise_regime)

    name = f"spy_phase1_{strategy}{keep_pct}_noise{noise_short}"
    if heteroscedastic:
        name += "_hetero"
    return name


def save_benchmark(
    df: pd.DataFrame,
    output_dir: Path,
    name: str,
    *,
    metadata: dict | None = None,
) -> Path:
    """Save a benchmark dataset as Parquet with embedded metadata.

    Parameters
    ----------
    df : pd.DataFrame
        The fully constructed benchmark dataset (with ``observed``,
        ``iv_clean``, ``split``, ``noise_sigma`` columns).
    output_dir : Path
        Directory to write into.
    name : str
        Benchmark name (used as filename stem).
    metadata : dict, optional
        Extra key-value pairs stored in the Parquet file metadata.

    Returns
    -------
    Path to the written file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"

    table = pa.Table.from_pandas(df, preserve_index=False)

    # Attach provenance metadata
    meta = {"benchmark_name": name}
    if metadata:
        meta.update({str(k): str(v) for k, v in metadata.items()})
    existing_meta = table.schema.metadata or {}
    existing_meta.update({k.encode(): v.encode() for k, v in meta.items()})
    table = table.replace_schema_metadata(existing_meta)

    pq.write_table(table, path)
    return path


def load_benchmark(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load a benchmark Parquet and return (DataFrame, metadata_dict)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = {}
    if table.schema.metadata:
        meta = {
            k.decode(): v.decode()
            for k, v in table.schema.metadata.items()
            if k != b"pandas"
        }
    return table.to_pandas(), meta
