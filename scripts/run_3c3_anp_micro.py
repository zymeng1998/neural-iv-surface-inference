#!/usr/bin/env python3
"""3C.3 single-config runner: ANP + ``micro_v1`` train + score + manifest.

Faithful restatement of ``scripts/run_3x9_anp.py`` (the OTM ANP baseline)
with the **only** modelling change being the per-quote feature tuple:
``data.feature_set: micro_v1`` (ADR 0008) extends the 3-dim context
``(log_moneyness, tau, iv)`` with six AV-native microstructure dims
``(bid, ask, bid_ask_spread_rel, volume, open_interest, put_call_indicator)``.

Normalization (ADR 0008 §Normalization, scope confirmed 2026-06-03):
    The five *new continuous* micro features
    ``(bid, ask, bid_ask_spread_rel, volume, open_interest)`` are z-scored
    against the OTM **train** split only; ``put_call_indicator`` (already in
    {-1, +1}) and the three legacy dims (fed raw in 3X.9) are left untouched.
    The fitted per-feature stats are persisted in the manifest so 3C.5+ /
    ``ConditionalSurfacePredictor`` can reproduce the input transform.

All z-scoring lives in *this driver* (3C.3 file_scope); the loader, encoder,
decoder, and ``train_conditional`` are imported unmodified (3C.2 contract).

Usage:
    python scripts/run_3c3_anp_micro.py \
        --config configs/conditional_3C3_anp_micro_gaussian_otm.yaml

Outputs (under config.paths.results_dir):
    checkpoints/best_conditional.pt   (Pod-side only — not pulled back)
    training_curve.csv
    predictions_val.parquet
    predictions_test.parquet
    manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from neural_iv_surface_inference.data.conditional_loaders import (
    _materialize_micro_columns,
    collate_conditional,
    feature_set_context_dim,
    resolve_context_features,
)
from neural_iv_surface_inference.models.conditional_surface import (
    ConditionalSurfaceModel,
)
from neural_iv_surface_inference.training.train_conditional import (
    train_conditional,
)
from neural_iv_surface_inference.utils.io import load_config

# Continuous micro features that get z-scored (ADR 0008; put_call_indicator
# and the 3 legacy dims are intentionally excluded — see module docstring).
_ZSCORE_COLS: tuple[str, ...] = (
    "bid",
    "ask",
    "bid_ask_spread_rel",
    "volume",
    "open_interest",
)
_STD_EPS = 1e-8
_QUERY_FEATURES = ("log_moneyness", "tau")


# ── Feature materialisation + z-score fit/apply (driver-owned) ─────────────


def _materialize(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Add the derived micro_v1 columns once (no-op for ``minimal``)."""
    if feature_set == "micro_v1":
        return _materialize_micro_columns(df)
    return df


def _fit_zscore_stats(
    train_df: pd.DataFrame, norm_cols: list[str]
) -> dict[str, dict[str, float]]:
    """Fit per-feature (mean, std) on the *train* split only."""
    stats: dict[str, dict[str, float]] = {}
    for col in norm_cols:
        x = pd.to_numeric(train_df[col], errors="coerce").to_numpy(np.float64)
        mean = float(np.nanmean(x))
        std = float(np.nanstd(x))
        stats[col] = {
            "mean": mean,
            "std": std,
            "std_effective": float(max(std, _STD_EPS)),
        }
    return stats


def _apply_zscore(
    df: pd.DataFrame, stats: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Return a copy with ``stats`` columns z-scored (train mean/std)."""
    out = df.copy()
    for col, s in stats.items():
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float64)
        out[col] = ((x - s["mean"]) / s["std_effective"]).astype(np.float32)
    return out


def _max_abs_z(df: pd.DataFrame, norm_cols: list[str]) -> dict[str, float]:
    """Diagnostic: largest |z| per normalized column (tail/outlier watch)."""
    return {
        col: float(np.nanmax(np.abs(pd.to_numeric(df[col], errors="coerce"))))
        for col in norm_cols
    }


# ── Pre-normalised date-grouped dataset (mirrors ConditionalIVSurfaceDataset
#    but consumes already-materialised + z-scored context columns; it must
#    NOT re-materialise, which would recompute bid_ask_spread_rel from the
#    z-scored bid/ask and corrupt it). ───────────────────────────────────────


class _PrenormConditionalDataset(Dataset):
    """Date-grouped dataset over a frame whose context columns are final.

    Faithful to ``ConditionalIVSurfaceDataset``'s grouping/query/target
    contract (stable date sort, observed-rows-as-context, all-rows-as-query,
    ``iv_clean`` target) but selects ``context_features`` verbatim instead of
    re-deriving the micro columns.
    """

    def __init__(self, df: pd.DataFrame, context_features: list[str]):
        self.context_features = list(context_features)
        required = set(self.context_features) | {
            "date",
            "observed",
            "iv_clean",
            *_QUERY_FEATURES,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"_PrenormConditionalDataset: missing columns {sorted(missing)}"
            )

        df = df.sort_values("date", kind="stable").reset_index(drop=True)
        self.dates: list[pd.Timestamp] = []
        self._contexts: list[torch.Tensor] = []
        self._queries: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._query_observed: list[torch.Tensor] = []

        for date, group in df.groupby("date", sort=True):
            obs_mask = group["observed"].values.astype(bool)
            if obs_mask.sum() == 0:
                raise ValueError(
                    f"_PrenormConditionalDataset: date {date} has zero observed "
                    "rows; the conditional model cannot form a context for it."
                )
            ctx_rows = group.loc[obs_mask, self.context_features].to_numpy(
                dtype=np.float32
            )
            query_rows = group[list(_QUERY_FEATURES)].to_numpy(dtype=np.float32)
            target_rows = group["iv_clean"].to_numpy(dtype=np.float32)

            self.dates.append(pd.Timestamp(date))
            self._contexts.append(torch.from_numpy(ctx_rows))
            self._queries.append(torch.from_numpy(query_rows))
            self._targets.append(torch.from_numpy(target_rows))
            self._query_observed.append(torch.from_numpy(obs_mask.astype(np.bool_)))

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        return {
            "context": self._contexts[idx],
            "query": self._queries[idx],
            "target": self._targets[idx],
            "query_observed": self._query_observed[idx],
            "date": self.dates[idx],
        }


def _build_loader(
    df: pd.DataFrame,
    context_features: list[str],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        _PrenormConditionalDataset(df, context_features),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_conditional,
        num_workers=0,
    )


# ── Misc helpers (cloned from run_3x9_anp.py) ──────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _cfg_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _build_model_from_cfg(cfg: dict, head_cfg: dict) -> ConditionalSurfaceModel:
    """Reconstruct the model from a persisted training config.

    ``feature_set`` is authoritative for the per-quote context dim, so
    ``context_dim`` is forced to ``None`` (passing the stale 3 would trip the
    3C.2 guard). Forwards decoder_kind / anp / coord_encoding so the ANP path
    round-trips through checkpoint reload.
    """
    return ConditionalSurfaceModel(
        context_dim=None,
        coord_dim=int(cfg.get("coord_dim", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        n_elem_layers=int(cfg.get("n_elem_layers", 2)),
        n_post_layers=int(cfg.get("n_post_layers", 1)),
        n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
        head=dict(head_cfg),
        coord_encoding=cfg.get("coord_encoding"),
        decoder_kind=str(cfg.get("decoder_kind", "deepsets")),  # type: ignore[arg-type]
        anp=cfg.get("anp"),
        feature_set=str(cfg.get("feature_set", "minimal")),
    )


@torch.no_grad()
def _score_split(
    model: ConditionalSurfaceModel,
    df: pd.DataFrame,
    head_kind: str,
    device: torch.device,
    context_features: list[str],
) -> pd.DataFrame:
    """Score a (materialised + z-scored) split frame; one row per query point.

    ``df`` must already carry the final ``context_features`` columns (the
    micro dims z-scored). Output prediction columns are all raw / untouched
    (``iv_true``, ``iv_clean``, coords) — none are normalized.
    """
    model.eval()
    n = len(df)
    mu_out = np.full(n, np.nan, dtype=np.float32)
    sigma_out = np.full(n, np.nan, dtype=np.float32)
    log_sigma2_out = np.full(n, np.nan, dtype=np.float32)
    q_lo = np.full(n, np.nan, dtype=np.float32)
    q_med = np.full(n, np.nan, dtype=np.float32)
    q_hi = np.full(n, np.nan, dtype=np.float32)

    df_indexed = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    for _date, group in df_indexed.groupby("date", sort=False):
        obs = group["observed"].values.astype(bool)
        if obs.sum() == 0:
            continue

        ctx = group.loc[obs, context_features].to_numpy(dtype=np.float32)
        query = group[list(_QUERY_FEATURES)].to_numpy(dtype=np.float32)
        idx = group["_orig_idx"].values

        ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)
        ctx_mask_t = torch.ones(1, ctx_t.shape[1], dtype=torch.bool, device=device)
        q_t = torch.from_numpy(query).unsqueeze(0).to(device)

        out = model(ctx_t, ctx_mask_t, q_t)
        if isinstance(out, dict):
            mu = out["mu"].squeeze(0).cpu().numpy()
            mu_out[idx] = mu
            if head_kind == "gaussian":
                sigma_out[idx] = out["sigma"].squeeze(0).cpu().numpy()
                log_sigma2_out[idx] = out["log_sigma2"].squeeze(0).cpu().numpy()
            elif head_kind == "quantile":
                q = out["quantiles"].squeeze(0).cpu().numpy()
                q_lo[idx] = q[:, 0]
                q_med[idx] = q[:, q.shape[1] // 2]
                q_hi[idx] = q[:, -1]
        else:
            mu_out[idx] = out.squeeze(0).cpu().numpy()

    pred_df = pd.DataFrame(
        {
            "date": df["date"].values,
            "log_moneyness": df["log_moneyness"].values.astype(np.float32),
            "tau": df["tau"].values.astype(np.float32),
            "observed": df["observed"].values.astype(bool),
            "iv_true": df["implied_volatility"].values.astype(np.float32),
            "iv_clean": df["iv_clean"].values.astype(np.float32)
            if "iv_clean" in df.columns
            else np.full(n, np.nan, dtype=np.float32),
            "mu": mu_out,
        }
    )
    if head_kind == "gaussian":
        pred_df["sigma"] = sigma_out
        pred_df["log_sigma2"] = log_sigma2_out
    elif head_kind == "quantile":
        pred_df["q_lo"] = q_lo
        pred_df["q_med"] = q_med
        pred_df["q_hi"] = q_hi

    return pred_df


def _mae(pred_df: pd.DataFrame, target_col: str = "iv_true") -> float:
    err = (pred_df["mu"] - pred_df[target_col]).abs()
    return float(err.dropna().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    config = load_config(args.config)
    data_cfg = dict(config.get("data", {}))
    feature_set = str(data_cfg.get("feature_set", "minimal"))
    context_features = list(resolve_context_features(feature_set))
    context_dim = feature_set_context_dim(feature_set)
    norm_cols = [c for c in context_features if c in _ZSCORE_COLS]

    cond_cfg = dict(config.get("conditional", {}))
    cond_cfg.setdefault("seed", config.get("seed", 42))
    # Inject feature_set so train_conditional builds the model with the right
    # input projection (context_dim) and persists it into the checkpoint.
    cond_cfg["feature_set"] = feature_set
    head_cfg = dict(cond_cfg.get("head", {"kind": "point"}))
    head_kind = str(head_cfg.get("kind", "point"))

    paths = config.get("paths", {})
    results_dir = Path(paths.get("results_dir", "artifacts/runs/3C3/_unknown"))
    checkpoint_dir = Path(paths.get("checkpoint_dir", results_dir / "checkpoints"))
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bench_file = Path(data_cfg.get("benchmark_file", ""))
    if not bench_file.exists():
        print(f"[3C.3] benchmark not found: {bench_file}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[3C.3] head_kind={head_kind}  feature_set={feature_set}  "
        f"context_dim={context_dim}  decoder_kind={cond_cfg.get('decoder_kind')}  "
        f"coord_encoding={cond_cfg.get('coord_encoding')}  "
        f"freeze_encoder={cond_cfg.get('freeze_encoder', False)}  "
        f"device={device}  benchmark={bench_file}"
    )
    print(f"[3C.3] context_features={context_features}")
    print(f"[3C.3] z-score columns (train-fit)={norm_cols}")

    df_all = pd.read_parquet(bench_file)
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
    print(
        f"[3C.3] rows: train={len(train_df):,}  val={len(val_df):,}  "
        f"test={len(test_df):,}"
    )

    # Materialise derived micro columns, then fit z-score stats on TRAIN only
    # and apply the same transform to every split (no leakage).
    train_m = _materialize(train_df, feature_set)
    val_m = _materialize(val_df, feature_set)
    test_m = _materialize(test_df, feature_set)

    zscore_stats = _fit_zscore_stats(train_m, norm_cols) if norm_cols else {}
    train_n = _apply_zscore(train_m, zscore_stats) if zscore_stats else train_m
    val_n = _apply_zscore(val_m, zscore_stats) if zscore_stats else val_m
    test_n = _apply_zscore(test_m, zscore_stats) if zscore_stats else test_m

    if zscore_stats:
        print("[3C.3] z-score stats (train-split):")
        for col, s in zscore_stats.items():
            print(
                f"          {col:>20s}  mean={s['mean']:.6g}  std={s['std']:.6g}"
            )
        z_diag = {
            "train": _max_abs_z(train_n, norm_cols),
            "test": _max_abs_z(test_n, norm_cols),
        }
        print(f"[3C.3] max |z| diagnostic: {z_diag}")
    else:
        z_diag = {}

    batch_size = int(data_cfg.get("batch_size", 32))
    train_loader = _build_loader(train_n, context_features, batch_size, shuffle=True)
    val_loader = _build_loader(val_n, context_features, batch_size, shuffle=False)

    t0 = time.time()
    model, history = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        config=cond_cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    train_elapsed = time.time() - t0
    print(f"[3C.3] training done in {train_elapsed:.1f}s  epochs={len(history)}")

    ckpt_path = checkpoint_dir / "best_conditional.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = _build_model_from_cfg(ckpt["config"], head_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    t_score = time.time()
    val_preds = _score_split(model, val_n, head_kind, device, context_features)
    test_preds = _score_split(model, test_n, head_kind, device, context_features)
    score_elapsed = time.time() - t_score

    val_path = results_dir / "predictions_val.parquet"
    test_path = results_dir / "predictions_test.parquet"
    val_preds.to_parquet(val_path, index=False)
    test_preds.to_parquet(test_path, index=False)

    curve = pd.DataFrame(history)
    curve_csv = results_dir / "training_curve.csv"
    curve.to_csv(curve_csv, index=False)

    val_mae = _mae(val_preds)
    test_mae = _mae(test_preds)
    final = history[-1]

    qmono_ok = True
    if head_kind == "quantile":
        qmono_ok = bool(
            (
                (test_preds["q_lo"] <= test_preds["q_med"])
                & (test_preds["q_med"] <= test_preds["q_hi"])
            )
            .dropna()
            .all()
        )

    manifest = {
        "story": "3C.3",
        "head_kind": head_kind,
        "feature_set": feature_set,
        "context_features": context_features,
        "context_dim": context_dim,
        "zscore_columns": norm_cols,
        "zscore_stats": zscore_stats,
        "zscore_max_abs_z": z_diag,
        "decoder_kind": str(cond_cfg.get("decoder_kind", "deepsets")),
        "anp": cond_cfg.get("anp"),
        "coord_encoding": cond_cfg.get("coord_encoding"),
        "freeze_encoder": bool(cond_cfg.get("freeze_encoder", False)),
        "seed": int(cond_cfg.get("seed", 42)),
        "git_sha": _git_sha(),
        "config_hash": _cfg_hash(cond_cfg),
        "benchmark_file": str(bench_file),
        "splits": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "epochs_completed": len(history),
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "best_val_loss": float(min(r["val_loss"] for r in history)),
        "val_mae_mu": val_mae,
        "test_mae_mu": test_mae,
        "quantile_monotonicity_ok": qmono_ok,
        "train_wall_clock_s": train_elapsed,
        "score_wall_clock_s": score_elapsed,
        "device": str(device),
        "outputs": {
            "checkpoint": str(ckpt_path),
            "predictions_val": str(val_path),
            "predictions_test": str(test_path),
            "training_curve": str(curve_csv),
        },
    }
    manifest_path = results_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(
        f"[3C.3] head={head_kind}  test_MAE_mu={test_mae:.6f}  "
        f"val_MAE_mu={val_mae:.6f}  qmono_ok={qmono_ok}  manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
