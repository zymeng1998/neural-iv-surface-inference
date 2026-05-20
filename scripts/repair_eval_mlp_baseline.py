#!/usr/bin/env python3
"""Phase 1 closeout repair — regenerate the missing ``mlp`` baseline rows.

EVAL-ONLY. This script loads the already-trained checkpoint
``artifacts/checkpoints/best_mlp.pt`` and evaluates it on the same benchmark
and splits used for the interpolation baseline, then merges the resulting
``mlp`` rows into ``artifacts/results/baseline_results.csv``.

It does **not**:
  - retrain the model (no optimizer, no backward pass),
  - modify or remove existing ``interp_rbf`` rows,
  - touch raw/processed parquet data.

Safety / correctness notes:
  - Existing ``mlp`` rows (if any) are dropped before appending, so re-running
    is idempotent.
  - All splits are evaluated with ``shuffle=False`` DataLoaders so that the
    concatenated predictions align row-for-row with the source DataFrame.
    (The original ``run_baseline.py`` evaluated the train split through the
    ``shuffle=True`` training loader, which misaligns predictions and rows;
    this repair avoids that.)

Usage:
    python scripts/repair_eval_mlp_baseline.py
    python scripts/repair_eval_mlp_baseline.py --benchmark path/to/benchmark.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.utils.io import load_config
from neural_iv_surface_inference.data.loaders import (
    IVSurfaceDataset,
    load_benchmark_splits,
)
from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.training.train import predict_mlp
from neural_iv_surface_inference.training.eval import (
    evaluate_predictions,
    metrics_to_dataframe,
    print_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-only MLP baseline repair")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--benchmark", default=None,
        help="Override benchmark parquet (default: config data.benchmark_file)",
    )
    parser.add_argument(
        "--checkpoint", default="artifacts/checkpoints/best_mlp.pt",
    )
    parser.add_argument(
        "--results", default="artifacts/results/baseline_results.csv",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    args = parser.parse_args()

    config = load_config(args.config)

    benchmark_path = Path(args.benchmark or config["data"]["benchmark_file"])
    if not benchmark_path.exists():
        print(f"Benchmark file not found: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint and rebuild model (NO training) ───────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mlp_cfg = ckpt.get("config") or config.get("mlp", {})
    model = BaselineMLP(
        input_dim=mlp_cfg.get("input_dim", 2),
        hidden_dim=mlp_cfg.get("hidden_dim", 128),
        n_layers=mlp_cfg.get("n_layers", 3),
        dropout=mlp_cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[repair] checkpoint={ckpt_path}  epoch={ckpt.get('epoch')}  "
          f"val_loss={ckpt.get('val_loss')}  device={device}")
    print(f"[repair] benchmark={benchmark_path}")

    # ── Load splits (raw DataFrames reused; loaders rebuilt shuffle=False) ──
    data = load_benchmark_splits(
        benchmark_path,
        batch_size=args.eval_batch_size,
        num_workers=0,
    )

    rows = []
    for split in ["train", "val", "test"]:
        df_split = data[f"{split}_df"].copy()
        loader = DataLoader(
            IVSurfaceDataset(df_split),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"\n[mlp eval] {split}: {len(df_split):,} points")
        df_split["iv_pred"] = predict_mlp(model, loader, device=device)

        results = evaluate_predictions(df_split)
        print_evaluation(results, model_name=f"MLP (eval-only) — {split}")

        summary = metrics_to_dataframe(results, model_name="mlp")
        summary["split"] = split
        rows.append(summary)

    mlp_df = pd.concat(rows, ignore_index=True)

    # ── Merge into results CSV without disturbing interp_rbf rows ──────
    results_path = Path(args.results)
    if results_path.exists():
        existing = pd.read_csv(results_path)
        kept = existing[existing["model"] != "mlp"]
        combined = pd.concat([kept, mlp_df], ignore_index=True)
    else:
        combined = mlp_df

    # Align columns with existing schema where possible
    combined.to_csv(results_path, index=False)
    print(f"\n[repair] wrote {results_path}")
    print(f"[repair] models now present: {sorted(combined['model'].unique())}")


if __name__ == "__main__":
    main()
