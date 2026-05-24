#!/usr/bin/env python3
"""Train a K-seed deep ensemble of the W3 conditional surface model (2D.3).

Each member is a full, independent invocation of the existing 2C.4 training
entrypoint (``train_conditional``) so disagreement reflects real
initialization + SGD variance.

Usage
-----

::

    python scripts/run_ensemble_train.py --config configs/conditional.yaml
    python scripts/run_ensemble_train.py --config configs/conditional.yaml \
        --ensemble-size 2 --smoke

Outputs
-------

Per seed::

    {checkpoint_dir}/ensemble/seed_{seed}/best_conditional.pt

Manifest at the parent directory::

    {checkpoint_dir}/ensemble/members.json

Manifest schema
---------------

::

    {
      "version": 1,
      "ensemble_size": <int>,
      "config_hash": "<sha256 prefix of resolved conditional config>",
      "members": [
        {"seed": <int>,
         "val_loss": <float>,
         "checkpoint": "seed_<seed>/best_conditional.pt"},
        ...
      ]
    }

The matching **remote** full-AV ensemble run on the benchmark dataset lives
in story 2D.8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


MANIFEST_VERSION = 1


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
        observed[-max(1, n_per_date // 4):] = False
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


def _hash_config(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--benchmark", type=Path, default=None,
        help="Override data.benchmark_file from the config.",
    )
    parser.add_argument(
        "--ensemble-size", type=int, default=None,
        help="Override ensemble.size from the config.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Train on a tiny synthetic dataset (no on-disk file needed).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cond_cfg = dict(config.get("conditional", {}))
    cond_cfg.setdefault("seed", config.get("seed", 42))

    ens_cfg = dict(config.get("ensemble", {}))
    size = int(args.ensemble_size if args.ensemble_size is not None
               else ens_cfg.get("size", 5))
    all_seeds = list(ens_cfg.get("seeds", []))
    if len(all_seeds) < size:
        raise ValueError(
            f"ensemble.seeds has {len(all_seeds)} entries but size={size}"
        )
    seeds = [int(s) for s in all_seeds[:size]]

    paths = config.get("paths", {})
    checkpoint_root = Path(paths.get("checkpoint_dir", "artifacts/checkpoints"))
    ensemble_dir = checkpoint_root / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ensemble] device={device}  size={size}  seeds={seeds}")

    if args.smoke:
        print("[ensemble] SMOKE MODE — synthetic data, short training")
        train_df = _synthetic_frame(n_dates=8, n_per_date=10, seed=0)
        val_df = _synthetic_frame(n_dates=4, n_per_date=10, seed=1)
        cond_cfg.update(
            hidden_dim=cond_cfg.get("smoke_hidden_dim", 16),
            latent_dim=cond_cfg.get("smoke_latent_dim", 8),
            epochs=cond_cfg.get("smoke_epochs", 6),
            patience=20,
        )
        batch_size = 4
    else:
        bench_file = args.benchmark or Path(
            config.get("data", {}).get("benchmark_file", "")
        )
        if not bench_file or not Path(bench_file).exists():
            print(
                f"[ensemble] benchmark file not found: {bench_file!r}",
                file=sys.stderr,
            )
            return 2
        print(f"[ensemble] benchmark: {bench_file}")
        df = pd.read_parquet(bench_file)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        batch_size = int(config.get("data", {}).get("batch_size", 32))

    train_loader = _build_loader(train_df, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_df, batch_size=batch_size, shuffle=False)

    cfg_hash = _hash_config(cond_cfg)

    members: list[dict] = []
    for seed in seeds:
        member_cfg = dict(cond_cfg)
        member_cfg["seed"] = seed
        member_dir = ensemble_dir / f"seed_{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        _, history = train_conditional(
            train_loader=train_loader,
            val_loader=val_loader,
            config=member_cfg,
            device=device,
            checkpoint_dir=member_dir,
        )
        elapsed = time.time() - t0

        final = history[-1]
        best_val = min(r["val_loss"] for r in history)
        print(
            f"[ensemble] seed={seed}  epochs={len(history)}  "
            f"final_train={final['train_loss']:.6f}  "
            f"final_val={final['val_loss']:.6f}  "
            f"best_val={best_val:.6f}  ({elapsed:.1f}s)"
        )
        members.append(
            {
                "seed": seed,
                "val_loss": float(best_val),
                "checkpoint": f"seed_{seed}/best_conditional.pt",
            }
        )

    manifest = {
        "version": MANIFEST_VERSION,
        "ensemble_size": len(members),
        "config_hash": cfg_hash,
        "members": members,
    }
    manifest_path = ensemble_dir / "members.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ensemble] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
