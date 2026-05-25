"""Fit the W4 calibrator on cached val predictions and verify on test (2D.4).

Reads val + test CSVs produced by 2D.7 (single-head full-AV runs) and
optionally 2D.8 (deep-ensemble), fits a
:class:`~neural_iv_surface_inference.eval.calibration.Calibrator`, persists it
to ``artifacts/calibration/<run_id>.json``, and prints W1 coverage +
error-uncertainty correlation on the held-out test fold. Pure CPU; no torch,
no model load, no AV data dependency at fit time.

Usage
-----

    python3 scripts/run_calibration_fit.py --config configs/calibration.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from neural_iv_surface_inference.eval.calibration import Calibrator
from neural_iv_surface_inference.eval.uncertainty_metrics import (
    error_uncertainty_correlation,
    interval_coverage,
    mean_interval_width,
)


JOIN_KEYS = ("date", "log_moneyness", "tau", "observed")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"prediction CSV not found: {path}")
    return pd.read_csv(path)


def _join_ensemble(
    base: pd.DataFrame, ensemble: pd.DataFrame
) -> pd.DataFrame:
    """Attach `disagreement_std` to ``base`` by positional alignment.

    The 2D.7 and 2D.8 evaluators both iterate the same dataset in the same
    order, so when row counts match we align by index. We then assert the join
    keys agree row-for-row before returning, to surface ordering bugs loudly.
    """
    if len(base) != len(ensemble):
        raise ValueError(
            f"row count mismatch: base={len(base)} ensemble={len(ensemble)} — "
            "cannot align ensemble disagreement positionally"
        )
    out = base.reset_index(drop=True).copy()
    ens = ensemble.reset_index(drop=True)
    # Cheap sanity check: a handful of rows should match on the join keys.
    sample = np.linspace(0, len(out) - 1, num=min(50, len(out)), dtype=int)
    for key in JOIN_KEYS:
        a = out.loc[sample, key].to_numpy()
        b = ens.loc[sample, key].to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(
                f"positional alignment check failed on column {key!r}; "
                "the 2D.7 and 2D.8 CSVs are not in the same row order"
            )
    out["disagreement_std"] = ens["disagreement_std"].to_numpy()
    return out


def _fit_calibrator(
    cfg: dict, val: pd.DataFrame, target_col: str
) -> Calibrator:
    head_kind = cfg["head_kind"]
    y = val[target_col].to_numpy(dtype=np.float64)
    mu = val["mu"].to_numpy(dtype=np.float64)
    disagreement = (
        val["disagreement_std"].to_numpy(dtype=np.float64)
        if "disagreement_std" in val.columns
        else None
    )
    if head_kind == "gaussian":
        sigma = val["sigma"].to_numpy(dtype=np.float64)
        return Calibrator.fit(
            head_kind="gaussian",
            y_true=y,
            mu=mu,
            sigma=sigma,
            disagreement=disagreement,
            nominal_alpha=cfg["nominal_alpha"],
            extra={"source": cfg["sources"]["gaussian"]["val"]},
        )
    if head_kind == "quantile":
        q_lo = val["q_lo"].to_numpy(dtype=np.float64)
        q_hi = val["q_hi"].to_numpy(dtype=np.float64)
        return Calibrator.fit(
            head_kind="quantile",
            y_true=y,
            mu=mu,
            q_lo=q_lo,
            q_hi=q_hi,
            disagreement=disagreement,
            nominal_alpha=cfg["nominal_alpha"],
            extra={"source": cfg["sources"]["quantile"]["val"]},
        )
    raise ValueError(f"unsupported head_kind: {head_kind!r}")


def _apply_calibrator(
    cal: Calibrator, frame: pd.DataFrame
) -> dict[str, np.ndarray]:
    head_kind = cal.fit_params.head_kind
    mu = frame["mu"].to_numpy(dtype=np.float64)
    kwargs: dict = {"mu": mu}
    if head_kind == "gaussian":
        kwargs["sigma"] = frame["sigma"].to_numpy(dtype=np.float64)
    else:
        kwargs["q_lo"] = frame["q_lo"].to_numpy(dtype=np.float64)
        kwargs["q_hi"] = frame["q_hi"].to_numpy(dtype=np.float64)
    if cal.fit_params.has_ensemble and "disagreement_std" in frame.columns:
        kwargs["disagreement"] = frame["disagreement_std"].to_numpy(
            dtype=np.float64
        )
    return cal.apply(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="path to calibration YAML"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    head_kind = cfg["head_kind"]
    src = cfg["sources"][head_kind]
    target_col = cfg.get("target_col", "iv_true")
    nominal = float(cfg["nominal_alpha"])
    tol = float(cfg.get("tolerance", 0.02))

    print(f"[1/5] loading val predictions: {src['val']}")
    val = _load_csv(Path(src["val"]))
    print(f"      val rows: {len(val):,}")

    print(f"[2/5] loading test predictions: {src['test']}")
    test = _load_csv(Path(src["test"]))
    print(f"      test rows: {len(test):,}")

    ens_cfg = cfg.get("ensemble") or {}
    if ens_cfg.get("val") and ens_cfg.get("test"):
        print(f"[3/5] joining ensemble disagreement from {ens_cfg['val']}")
        ens_val = _load_csv(Path(ens_cfg["val"]))
        ens_test = _load_csv(Path(ens_cfg["test"]))
        val = _join_ensemble(val, ens_val)
        test = _join_ensemble(test, ens_test)
    else:
        print("[3/5] no ensemble source configured — fusion uses primary head only")

    print(f"[4/5] fitting calibrator (head={head_kind}, alpha={nominal})")
    cal = _fit_calibrator(cfg, val, target_col)
    out_path = Path(cfg["output_path"])
    cal.save(out_path)
    print(f"      wrote {out_path}")
    print(
        "      fit params: "
        + json.dumps({k: v for k, v in cal.to_dict()["fit"].items() if k != "extra"})
    )

    print("[5/5] verifying on held-out test fold")
    out_test = _apply_calibrator(cal, test)
    y_test = test[target_col].to_numpy(dtype=np.float64)
    abs_err = np.abs(y_test - out_test["pred"])
    cov = interval_coverage(y_test, out_test["lower"], out_test["upper"])
    width = mean_interval_width(out_test["lower"], out_test["upper"])
    corr = error_uncertainty_correlation(abs_err, out_test["uncertainty"])

    report = {
        "head_kind": head_kind,
        "nominal_alpha": nominal,
        "test_coverage": cov,
        "test_mean_interval_width": width,
        "test_error_uncertainty_corr_pearson": corr["pearson"],
        "test_error_uncertainty_corr_spearman": corr["spearman"],
        "test_n": int(len(y_test)),
        "tolerance": tol,
        "within_tolerance": bool(abs(cov - nominal) <= tol),
    }
    print(json.dumps(report, indent=2))

    sidecar = out_path.with_suffix(".report.json")
    sidecar.write_text(json.dumps(report, indent=2))
    print(f"      wrote {sidecar}")

    if not report["within_tolerance"]:
        print(
            f"[fail] test coverage {cov:.4f} outside ±{tol} of nominal {nominal}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
