#!/usr/bin/env python3
"""Train the W3 conditional surface model end-to-end (2C.4).

Usage:
    python scripts/run_conditional.py --config configs/conditional.yaml
    python scripts/run_conditional.py --config configs/conditional.yaml --smoke
    python scripts/run_conditional.py --config configs/conditional.yaml --benchmark path/to/file.parquet

The ``--smoke`` flag bypasses the on-disk benchmark and trains for a few epochs
on a tiny synthetic frame, useful for local CI / wiring validation. Verbose
training output should be redirected to a log file by the caller (per
CLAUDE.md context rules) — this script prints only summary lines.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    collate_conditional,
)
from neural_iv_surface_inference.training.train_conditional import (
    train_conditional,
)
from neural_iv_surface_inference.utils.io import load_config


def _build_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    ds = ConditionalIVSurfaceDataset(df)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_conditional,
        num_workers=0,
    )


def _synthetic_frame(n_dates: int, n_per_date: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_dates):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        k = rng.uniform(-0.3, 0.3, n_per_date).astype(np.float32)
        tau = rng.uniform(0.05, 1.0, n_per_date).astype(np.float32)
        iv_clean = (0.2 + 0.5 * k**2 + 0.1 * tau).astype(np.float32)
        iv_noisy = iv_clean + rng.normal(0, 0.005, n_per_date).astype(np.float32)
        observed = np.ones(n_per_date, dtype=bool)
        observed[-max(1, n_per_date // 4) :] = False
        parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "log_moneyness": k,
                    "tau": tau,
                    "implied_volatility": iv_noisy,
                    "iv_clean": iv_clean,
                    "observed": observed,
                    "split": "train",
                    "noise_sigma": 0.005,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Override data.benchmark_file from the config.")
    parser.add_argument("--smoke", action="store_true",
                        help="Train on a tiny synthetic dataset (no on-disk file needed).")
    args = parser.parse_args()

    config = load_config(args.config)
    cond_cfg = dict(config.get("conditional", {}))
    cond_cfg.setdefault("seed", config.get("seed", 42))

    paths = config.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoint_dir", "artifacts/checkpoints"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[conditional] device={device}")

    if args.smoke:
        print("[conditional] SMOKE MODE — synthetic data, short training")
        train_df = _synthetic_frame(n_dates=8, n_per_date=10, seed=0)
        val_df = _synthetic_frame(n_dates=4, n_per_date=10, seed=1)
        # Lighten the model for the smoke run
        cond_cfg.update(
            hidden_dim=cond_cfg.get("smoke_hidden_dim", 16),
            latent_dim=cond_cfg.get("smoke_latent_dim", 8),
            epochs=cond_cfg.get("smoke_epochs", 8),
            patience=20,
        )
        batch_size = 4
    else:
        bench_file = args.benchmark or Path(
            config.get("data", {}).get("benchmark_file", "")
        )
        if not bench_file or not Path(bench_file).exists():
            print(f"[conditional] benchmark file not found: {bench_file!r}", file=sys.stderr)
            return 2
        print(f"[conditional] benchmark: {bench_file}")
        df = pd.read_parquet(bench_file)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        batch_size = int(config.get("data", {}).get("batch_size", 32))

    train_loader = _build_loader(train_df, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_df, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    model, history = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cond_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    elapsed = time.time() - t0

    final = history[-1]
    print(
        f"[conditional] done in {elapsed:.1f}s  epochs={len(history)}  "
        f"final_train_loss={final['train_loss']:.6f}  "
        f"final_val_loss={final['val_loss']:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
