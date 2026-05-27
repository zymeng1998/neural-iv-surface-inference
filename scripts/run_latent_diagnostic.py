#!/usr/bin/env python3
"""Latent capacity diagnostic for a ``ConditionalSurfaceModel`` (Story 2E.2).

End-to-end runner: load config + checkpoint, extract per-date latents from the
encoder over the chosen split, compute the SVD spectrum, and run per-dim and
per-PC ablations against the decoder. Emits the artifact bundle listed in the
2E.2 spec (``rank_report.json``, three CSVs, four PNGs, the raw latents, and
a run log).

Local note: this runner is callable on a Mac but only meaningful on the Pod
because the benchmark parquet and production checkpoint live there. The
intermediate diagnostic modules are unit-tested locally.

Usage
-----
::

    python scripts/run_latent_diagnostic.py \\
        --config configs/conditional_2D7_gaussian.yaml \\
        --checkpoint artifacts/runs/2D7/gaussian/checkpoints/best_conditional.pt \\
        --split val \\
        --out artifacts/diagnostics/2E2/prod_2d7_gaussian/

Verbose tensor / matplotlib output is redirected to ``<out>/run.log``; only
summary scalars print to stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

# Make src/ importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neural_iv_surface_inference.data.conditional_loaders import (
    ConditionalIVSurfaceDataset,
    collate_conditional,
)
from neural_iv_surface_inference.diagnostics.contribution import (
    LossFn,
    ablate_dim,
    ablate_pc,
    baseline_loss,
    topk_pc_reconstruction,
)
from neural_iv_surface_inference.diagnostics.effective_rank import (
    RankReport,
    analyze,
)
from neural_iv_surface_inference.diagnostics.latent_probe import (
    LatentCache,
    extract_latents,
)
from neural_iv_surface_inference.eval.adapters import (
    ConditionalSurfacePredictor,
)
from neural_iv_surface_inference.utils.io import load_config


TOPK_GRID: tuple[int, ...] = (1, 2, 3, 5, 8, 16, 32, 64)
_LOG = logging.getLogger("latent_diagnostic")


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)


def _build_split_loader(
    config: dict,
    split: str,
    batch_size: int,
    max_dates: int | None = None,
    sample_seed: int = 42,
) -> DataLoader:
    bench_file = config.get("data", {}).get("benchmark_file")
    if not bench_file or not Path(bench_file).exists():
        raise FileNotFoundError(
            f"benchmark file not found: {bench_file!r} "
            "(local trees do not carry the AV parquet; run on the Pod)"
        )
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test'; got {split!r}")
    # Filter at parquet read time. The benchmark parquets are ~1 GB on disk
    # and balloon to several GB in pandas memory; loading only the target
    # split keeps the runner within the Pod's 8 GB container limit.
    sub = pd.read_parquet(bench_file, filters=[("split", "=", split)]).reset_index(
        drop=True
    )
    if sub.empty:
        raise RuntimeError(f"split {split!r} is empty in {bench_file}")

    # On Pods with a tight memory limit (8 GB container), the full val split
    # of the AV benchmark (~2.8 GB pandas + dataset duplication) OOMs during
    # dataset construction. ``--max-dates`` samples a deterministic subset
    # of dates so the diagnostic still produces a representative SVD spectrum
    # and ablation grid (N >> latent_dim=64 is the only constraint).
    if max_dates is not None and max_dates > 0:
        all_dates = np.array(sorted(sub["date"].unique()))
        if max_dates < len(all_dates):
            rng = np.random.default_rng(sample_seed)
            picked = np.sort(rng.choice(all_dates, size=max_dates, replace=False))
            sub = sub[sub["date"].isin(picked)].reset_index(drop=True)
            _LOG.info(
                "sampled %d / %d dates (seed=%d) -> %d rows",
                len(picked), len(all_dates), sample_seed, len(sub),
            )

    ds = ConditionalIVSurfaceDataset(sub)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_conditional,
        num_workers=0,
    )
    # Drop the intermediate dataframe — the dataset has internalized what it
    # needs and ``sub`` is the biggest single allocation. Saves ~2-3 GB.
    del sub
    return loader


def _build_loss_fn(
    model: torch.nn.Module,
    cache: LatentCache,
    head_kind: str,
    device: torch.device,
) -> LossFn:
    """Bind decoder + cached per-row payload into a per-row loss function.

    The returned callable takes ``(Z_batch, row_indices)`` and returns a
    1-D tensor of per-row losses. Each row's loss is the loss-function mean
    over its (unpadded) query points — matching the per-row aggregation that
    ablation utilities then average over.
    """
    decoder = model.decoder
    target_all = cache.target
    query_all = cache.query
    mask_all = cache.query_mask
    eps = 1e-6

    if head_kind not in {"point", "gaussian"}:
        # 2E.2 targets the 2D.7 gaussian checkpoint. Quantile is a separate
        # follow-up (it would need pinball aggregation here); reject early
        # rather than silently misreport.
        raise NotImplementedError(
            f"latent diagnostic only supports point|gaussian heads; "
            f"got head_kind={head_kind!r}"
        )

    @torch.no_grad()
    def loss_fn(Z_batch: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        q = query_all[idx].to(device)
        t = target_all[idx].to(device)
        m = mask_all[idx].to(device)
        Zb = Z_batch.to(device)
        out = decoder(Zb, q)
        if head_kind == "point":
            mu = out  # CoordinateDecoder returns a [B, Q] tensor directly
            per_point = (mu - t) ** 2
        else:
            mu = out["mu"]
            sigma = out["sigma"].clamp_min(eps)
            log_sigma2 = 2.0 * torch.log(sigma)
            per_point = 0.5 * (log_sigma2 + ((t - mu) ** 2) / (sigma ** 2))
        mask_f = m.to(per_point.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        per_row = (per_point * mask_f).sum(dim=1) / denom
        return per_row.detach().cpu()

    return loss_fn


def _write_singular_values(report: RankReport, path: Path) -> None:
    df = pd.DataFrame(
        {
            "idx": np.arange(report.singular_values.shape[0]),
            "sigma": report.singular_values,
            "sigma_squared": report.singular_values ** 2,
            "variance_ratio": report.variance_ratio,
            "cumulative": report.cumulative_variance,
        }
    )
    df.to_csv(path, index=False)


def _write_pc_loadings(report: RankReport, csv_path: Path, npy_path: Path) -> None:
    np.save(npy_path, report.pc_loadings)
    d = report.pc_loadings.shape[0]
    rows = []
    for pc in range(d):
        col = report.pc_loadings[:, pc]
        for raw in range(d):
            val = float(col[raw])
            if abs(val) >= 0.05:
                rows.append({"pc_idx": pc, "raw_dim_idx": raw, "loading": val})
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_rank_report_json(report: RankReport, path: Path) -> None:
    payload = {
        "singular_values": report.singular_values.tolist(),
        "variance_ratio": report.variance_ratio.tolist(),
        "cumulative_variance": report.cumulative_variance.tolist(),
        "eff_rank_entropy": report.eff_rank_entropy,
        "stable_rank": report.stable_rank,
        "k95": report.k95,
        "k99": report.k99,
        "dead_pcs": report.dead_pcs,
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _maybe_make_plots(
    report: RankReport,
    per_dim_csv: pd.DataFrame,
    per_pc_csv: pd.DataFrame,
    topk_csv: pd.DataFrame,
    out_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _LOG.warning("matplotlib not available; skipping PNG plots")
        return

    d = report.singular_values.shape[0]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(np.arange(1, d + 1), report.singular_values ** 2, marker="o")
    ax.set_xlabel("PC index")
    ax.set_ylabel(r"$\sigma_i^2$ (log)")
    ax.set_title("Singular spectrum")
    fig.tight_layout()
    fig.savefig(out_dir / "singular_spectrum.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(
        np.arange(1, d + 1), report.cumulative_variance, marker="o"
    )
    ax.axhline(0.95, color="grey", linestyle="--", linewidth=0.8, label="0.95")
    ax.axhline(0.99, color="grey", linestyle=":", linewidth=0.8, label="0.99")
    ax.set_xlabel("k (number of PCs)")
    ax.set_ylabel("cumulative variance ratio")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Cumulative variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_variance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    dims = per_dim_csv.sort_values("dim")
    ax.bar(dims["dim"], dims["val_nll_delta"])
    ax.set_xlabel("latent dim index")
    ax.set_ylabel(r"$\Delta$ loss vs baseline")
    ax.set_title("Per-dim mean-substitution ablation")
    fig.tight_layout()
    fig.savefig(out_dir / "per_dim_ablation.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    pcs = per_pc_csv.sort_values("pc")
    ax.bar(pcs["pc"], pcs["val_nll_delta"], label=r"$\Delta$ loss")
    ax2 = ax.twinx()
    ax2.plot(
        pcs["pc"], pcs["variance_ratio"], color="orange",
        marker=".", linewidth=1.0, label="variance ratio",
    )
    ax.set_xlabel("PC index")
    ax.set_ylabel(r"$\Delta$ loss")
    ax2.set_ylabel("variance ratio")
    ax.set_title("Per-PC ablation vs variance share")
    fig.tight_layout()
    fig.savefig(out_dir / "per_pc_ablation.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    topk = topk_csv.sort_values("k")
    ax.plot(topk["k"], topk["val_nll_delta"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("k (top-k PCs kept)")
    ax.set_ylabel(r"$\Delta$ loss vs baseline")
    ax.set_title("Top-k PC reconstruction loss")
    fig.tight_layout()
    fig.savefig(out_dir / "topk_pc_reconstruction.png", dpi=130)
    plt.close(fig)


def _topk_grid(latent_dim: int) -> Iterable[int]:
    return [k for k in TOPK_GRID if k <= latent_dim]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default=None,
        help="torch device override (defaults to cuda if available, else cpu)",
    )
    parser.add_argument(
        "--max-dates", type=int, default=None,
        help=(
            "Cap the number of distinct dates loaded from the split (deterministic"
            " random sample, seed=--sample-seed). Use on memory-constrained Pods"
            " where the full val/test split OOMs during dataset construction."
        ),
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Seed for --max-dates sampling. Has no effect when --max-dates is unset.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(out_dir / "run.log")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _LOG.info("device=%s", device)

    config = load_config(args.config)
    loader = _build_split_loader(
        config,
        split=args.split,
        batch_size=args.batch_size,
        max_dates=args.max_dates,
        sample_seed=args.sample_seed,
    )
    _LOG.info("built %s loader: %d batches", args.split, len(loader))

    predictor = ConditionalSurfacePredictor.from_checkpoint(
        args.checkpoint, device=device
    )
    model = predictor.model
    head_kind = getattr(model, "head_kind", "point")
    _LOG.info("loaded checkpoint head_kind=%s latent_dim=%d", head_kind,
              model.encoder.post_mlp[-1].out_features)

    t0 = time.time()
    cache = extract_latents(model, loader, device=device)
    _LOG.info("extracted Z shape=%s in %.1fs", tuple(cache.Z.shape), time.time() - t0)
    # Free the dataset + loader; ablation steps work entirely from the cache.
    del loader

    Z_np = cache.Z.numpy().astype(np.float64)
    report = analyze(Z_np)

    # Save raw latents + report scalars.
    np.save(out_dir / "latents.npy", Z_np)
    _write_rank_report_json(report, out_dir / "rank_report.json")
    _write_singular_values(report, out_dir / "singular_values.csv")
    _write_pc_loadings(
        report,
        csv_path=out_dir / "pc_loadings.csv",
        npy_path=out_dir / "pc_loadings.npy",
    )

    loss_fn = _build_loss_fn(model, cache, head_kind=head_kind, device=device)
    baseline = baseline_loss(loss_fn, cache.Z, batch_size=args.batch_size)
    _LOG.info("baseline %s loss = %.6f", head_kind, baseline)

    d = cache.Z.shape[1]

    t0 = time.time()
    per_dim_delta = np.array(
        [
            ablate_dim(loss_fn, cache.Z, dim=i, batch_size=args.batch_size)
            - baseline
            for i in range(d)
        ]
    )
    _LOG.info("per-dim ablation done in %.1fs", time.time() - t0)

    per_dim_df = pd.DataFrame(
        {
            "dim": np.arange(d),
            "val_nll_delta": per_dim_delta,
        }
    )
    per_dim_df["abs_rank"] = (
        per_dim_df["val_nll_delta"].abs().rank(ascending=False, method="min").astype(int)
    )
    per_dim_df.sort_values("abs_rank").to_csv(
        out_dir / "per_dim_ablation.csv", index=False
    )

    pc_loadings_t = torch.tensor(report.pc_loadings, dtype=cache.Z.dtype)
    Z_mean = cache.Z.mean(dim=0)

    t0 = time.time()
    per_pc_delta = np.array(
        [
            ablate_pc(
                loss_fn, cache.Z, pc_idx=i,
                pc_loadings=pc_loadings_t, Z_mean=Z_mean,
                batch_size=args.batch_size,
            )
            - baseline
            for i in range(d)
        ]
    )
    _LOG.info("per-PC ablation done in %.1fs", time.time() - t0)

    per_pc_df = pd.DataFrame(
        {
            "pc": np.arange(d),
            "val_nll_delta": per_pc_delta,
            "variance_ratio": report.variance_ratio,
        }
    )
    per_pc_df["abs_rank"] = (
        per_pc_df["val_nll_delta"].abs().rank(ascending=False, method="min").astype(int)
    )
    per_pc_df.sort_values("abs_rank").to_csv(
        out_dir / "per_pc_ablation.csv", index=False
    )

    t0 = time.time()
    grid = list(_topk_grid(d))
    topk_rows = []
    for k in grid:
        l = topk_pc_reconstruction(
            loss_fn, cache.Z, k=k, pc_loadings=pc_loadings_t,
            Z_mean=Z_mean, batch_size=args.batch_size,
        )
        delta = l - baseline
        topk_rows.append(
            {
                "k": k,
                "val_nll_delta": delta,
                "fraction_of_baseline": (l - baseline) / max(abs(baseline), 1e-12),
            }
        )
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(out_dir / "topk_pc_reconstruction.csv", index=False)
    _LOG.info("top-k reconstruction done in %.1fs", time.time() - t0)

    with contextlib.suppress(Exception):
        _maybe_make_plots(report, per_dim_df, per_pc_df, topk_df, out_dir)

    # Summary to stdout — short, per CLAUDE.md §2.3.
    print(
        f"[2E.2] split={args.split} N={cache.Z.shape[0]} latent_dim={d}\n"
        f"       eff_rank_entropy={report.eff_rank_entropy:.3f}  "
        f"stable_rank={report.stable_rank:.3f}  "
        f"k95={report.k95}  k99={report.k99}  dead_pcs={report.dead_pcs}\n"
        f"       baseline_{head_kind}_loss={baseline:.6f}\n"
        f"       top-3 most-sensitive dims: "
        f"{per_dim_df.nlargest(3, 'val_nll_delta')['dim'].tolist()}\n"
        f"       top-3 most-sensitive PCs : "
        f"{per_pc_df.nlargest(3, 'val_nll_delta')['pc'].tolist()}\n"
        f"       top-k Δloss curve         : "
        f"{[(int(r['k']), round(float(r['val_nll_delta']), 4)) for _, r in topk_df.iterrows()]}\n"
        f"       artifacts -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
