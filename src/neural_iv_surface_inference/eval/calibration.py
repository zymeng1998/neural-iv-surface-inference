"""Calibration + uncertainty-fusion for the W4 conditional predictor (2D.4).

Pure NumPy / SciPy. Fits on **cached validation predictions** (no model
re-training, no AV data dependency at fit time) and emits a serialisable
:class:`Calibrator` blob. Consumed by :class:`CalibratedConditionalPredictor`
in :mod:`neural_iv_surface_inference.eval.adapters`.

Three calibration primitives:

1. **Temperature scaling** (Gaussian head from 2D.2). Find ``T > 0`` such that
   the empirical coverage of ``mu +/- z_alpha * T * sigma`` on val equals the
   nominal level ``alpha``. Strictly monotone in ``T``, so a bisection on the
   coverage-vs-T curve is exact.
2. **Split-conformal adjustment** (quantile head from 2D.2). Add a single scalar
   ``delta`` to widen / narrow ``(q_lo, q_hi)`` so the val empirical coverage
   matches ``alpha``. ``delta = quantile(scores, ceil((n+1)*alpha)/n)`` with
   ``score_i = max(q_lo_i - y_i, y_i - q_hi_i)`` — positive when ``y`` is outside
   the interval, negative when inside.
3. **Monotone scaling for an auxiliary uncertainty source** (deep-ensemble
   disagreement from 2D.3 and the 2B.2 masking-sensitivity std). A simple
   non-negative-slope least-squares fit maps the raw signal into the same
   sigma-units as the primary head: ``sigma_eq = max(0, a*src + b)``.

Fusion (single per-point ``confidence_score`` and a fused ``uncertainty``)
combines the calibrated sigma-unit signals by **quadrature sum**:

    u = sqrt(sigma_primary**2 + sigma_eq_ensemble**2 + sigma_eq_masking**2)

which is monotone non-decreasing in each non-negative source. The
confidence score is a sigmoid of the negated, location-scale-normalised ``u``:

    confidence = sigmoid(-(u - u0) / s),    (u0, s) fit on val.

Higher ``u`` -> lower confidence. The ``(lower, upper)`` band comes from the
**primary** head (temperature-scaled Gaussian or conformalised quantile);
fusion is reserved for the ranking signal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Low-level fitting primitives
# ---------------------------------------------------------------------------


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def gaussian_z(alpha: float) -> float:
    """Two-sided central z-score for a Gaussian interval at nominal ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    return float(stats.norm.ppf(0.5 + alpha / 2.0))


def fit_temperature_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: float = 0.9,
    t_lo: float = 1e-3,
    t_hi: float = 1e3,
    tol: float = 1e-4,
) -> float:
    """Bisection on ``T > 0`` so coverage(mu +- z_alpha * T * sigma) == alpha.

    Coverage is monotone non-decreasing in ``T`` for fixed ``mu``, ``sigma``,
    so the bisection is well-defined. If the empirical coverage saturates
    below ``alpha`` at ``t_hi`` we return ``t_hi`` (best we can do at the given
    interval spec); analogously at ``t_lo``.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = _finite_mask(y, mu, sigma) & (sigma > 0)
    if mask.sum() < 2:
        raise ValueError("temperature fit needs >=2 finite (y, mu, sigma>0) rows")
    y, mu, sigma = y[mask], mu[mask], sigma[mask]
    z = gaussian_z(alpha)
    residual = np.abs(y - mu) / sigma

    def coverage(t: float) -> float:
        return float(np.mean(residual <= z * t))

    if coverage(t_hi) < alpha:
        return float(t_hi)
    if coverage(t_lo) >= alpha:
        return float(t_lo)
    lo, hi = t_lo, t_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if coverage(mid) >= alpha:
            hi = mid
        else:
            lo = mid
    return float(hi)


def fit_conformal_quantile(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    alpha: float = 0.9,
) -> float:
    """Split-conformal delta widening (q_lo - delta, q_hi + delta) on val.

    ``score_i = max(q_lo_i - y_i, y_i - q_hi_i)`` — positive when ``y`` falls
    outside ``[q_lo, q_hi]``. ``delta = quantile(scores, ceil((n+1)*alpha)/n)``,
    capped at 1 to handle the degenerate ``alpha`` close to 1, ``n`` small case.
    Returns the scalar widening; can be negative if the raw interval was already
    over-covering on val.
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    mask = _finite_mask(y, lo, hi)
    if mask.sum() < 2:
        raise ValueError("conformal fit needs >=2 finite (y, q_lo, q_hi) rows")
    y, lo, hi = y[mask], lo[mask], hi[mask]
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * alpha) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_monotone_scaling(
    abs_error: np.ndarray,
    source: np.ndarray,
) -> tuple[float, float]:
    """Non-negative-slope linear fit ``|err| ~= a * source + b``.

    Used to map an auxiliary uncertainty signal (deep-ensemble disagreement,
    masking-sensitivity std) into the same sigma-units as the primary head.
    Slope is clipped to non-negative so the mapping is monotone non-decreasing
    in the source. Returns ``(a, b)``.
    """
    ae = np.asarray(abs_error, dtype=float)
    src = np.asarray(source, dtype=float)
    mask = _finite_mask(ae, src)
    if mask.sum() < 2:
        return (0.0, float(np.nanmean(ae)) if mask.any() else 0.0)
    ae, src = ae[mask], src[mask]
    src_var = float(src.var())
    if src_var <= 0.0:
        return (0.0, float(ae.mean()))
    cov = float(np.cov(src, ae, ddof=0)[0, 1])
    a = max(0.0, cov / src_var)
    b = float(ae.mean() - a * src.mean())
    return (a, b)


def apply_monotone_scaling(
    source: np.ndarray, a: float, b: float
) -> np.ndarray:
    """Apply ``sigma_eq = max(0, a * source + b)`` element-wise."""
    src = np.asarray(source, dtype=float)
    out = a * src + b
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# Fusion + confidence
# ---------------------------------------------------------------------------


def fuse_uncertainty(
    sigma_primary: np.ndarray,
    sigma_ensemble: np.ndarray | None = None,
    sigma_masking: np.ndarray | None = None,
) -> np.ndarray:
    """Quadrature sum of non-negative sigma-unit signals.

    NaNs in any contributor are treated as zero (the source is unavailable for
    that row); the result is NaN only if the primary sigma is NaN.
    """
    primary = np.asarray(sigma_primary, dtype=float)
    parts = [primary**2]
    for aux in (sigma_ensemble, sigma_masking):
        if aux is None:
            continue
        a = np.asarray(aux, dtype=float)
        a = np.where(np.isfinite(a), a, 0.0)
        parts.append(a**2)
    return np.sqrt(np.sum(parts, axis=0))


def fit_confidence_params(u: np.ndarray) -> tuple[float, float]:
    """Centre / scale for the sigmoid mapping confidence = sigmoid(-(u-u0)/s).

    Uses the median and a robust scale (MAD / 0.6745) so the mapping is
    insensitive to heavy tails in the fused uncertainty distribution.
    """
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    if len(u) < 2:
        return (0.0, 1.0)
    u0 = float(np.median(u))
    mad = float(np.median(np.abs(u - u0)))
    s = mad / 0.6745 if mad > 0 else float(u.std() or 1.0)
    return (u0, max(s, 1e-12))


def confidence_from_u(u: np.ndarray, u0: float, s: float) -> np.ndarray:
    """Sigmoid mapping: higher fused uncertainty -> lower confidence."""
    u = np.asarray(u, dtype=float)
    z = -(u - u0) / s
    # Numerically stable sigmoid.
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return out


# ---------------------------------------------------------------------------
# Calibrator (fit + apply + serialise)
# ---------------------------------------------------------------------------


HeadKind = Literal["gaussian", "quantile"]


@dataclass(frozen=True)
class CalibrationFit:
    """Serialisable fitted-calibrator parameters.

    Either ``temperature`` (gaussian head) or ``conformal_delta`` (quantile
    head) is populated; the other is ``None``. Auxiliary signals always carry
    a ``(scale, bias)`` pair; when an aux source was not provided at fit time
    its pair is ``(0.0, 0.0)`` (so its contribution to fusion is zero).
    """

    head_kind: HeadKind
    nominal_alpha: float
    temperature: float | None = None
    conformal_delta: float | None = None
    ensemble_scale: float = 0.0
    ensemble_bias: float = 0.0
    masking_scale: float = 0.0
    masking_bias: float = 0.0
    has_ensemble: bool = False
    has_masking: bool = False
    u0: float = 0.0
    u_scale: float = 1.0
    n_fit: int = 0
    extra: dict = field(default_factory=dict)


class Calibrator:
    """Fit-once / apply-many calibrator for the W4 uncertainty stack.

    Construct via :meth:`fit` from cached val predictions; persist via
    :meth:`save` / :meth:`load` for re-use at test time.
    """

    def __init__(self, fit: CalibrationFit) -> None:
        self.fit_params = fit
        self.z_alpha = gaussian_z(fit.nominal_alpha)

    # ----- fitting -----

    @classmethod
    def fit(
        cls,
        *,
        head_kind: HeadKind,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray | None = None,
        q_lo: np.ndarray | None = None,
        q_hi: np.ndarray | None = None,
        disagreement: np.ndarray | None = None,
        masking_std: np.ndarray | None = None,
        nominal_alpha: float = 0.9,
        extra: dict | None = None,
    ) -> "Calibrator":
        if head_kind not in {"gaussian", "quantile"}:
            raise ValueError(f"unsupported head_kind: {head_kind!r}")
        if head_kind == "gaussian":
            if sigma is None:
                raise ValueError("gaussian fit requires sigma")
            temperature = fit_temperature_gaussian(
                y_true, mu, sigma, alpha=nominal_alpha
            )
            conformal_delta = None
            # primary-sigma for aux fitting and fusion: temperature-scaled.
            primary_sigma_val = float(temperature) * np.asarray(sigma, dtype=float)
        else:
            if q_lo is None or q_hi is None:
                raise ValueError("quantile fit requires q_lo and q_hi")
            temperature = None
            conformal_delta = fit_conformal_quantile(
                y_true, q_lo, q_hi, alpha=nominal_alpha
            )
            # Use half-width of the conformalised interval as primary sigma
            # surrogate (in the same units as |error|): (q_hi - q_lo + 2*delta)/2.
            primary_sigma_val = (
                np.asarray(q_hi, dtype=float)
                - np.asarray(q_lo, dtype=float)
                + 2.0 * conformal_delta
            ) / 2.0

        abs_err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float))

        ensemble_scale = ensemble_bias = 0.0
        masking_scale = masking_bias = 0.0
        has_ens = disagreement is not None
        has_msk = masking_std is not None
        if has_ens:
            ensemble_scale, ensemble_bias = fit_monotone_scaling(
                abs_err, disagreement
            )
        if has_msk:
            masking_scale, masking_bias = fit_monotone_scaling(
                abs_err, masking_std
            )

        sigma_ens_val = (
            apply_monotone_scaling(disagreement, ensemble_scale, ensemble_bias)
            if has_ens
            else None
        )
        sigma_msk_val = (
            apply_monotone_scaling(masking_std, masking_scale, masking_bias)
            if has_msk
            else None
        )

        u_val = fuse_uncertainty(primary_sigma_val, sigma_ens_val, sigma_msk_val)
        u0, s = fit_confidence_params(u_val)

        fit_obj = CalibrationFit(
            head_kind=head_kind,
            nominal_alpha=float(nominal_alpha),
            temperature=float(temperature) if temperature is not None else None,
            conformal_delta=(
                float(conformal_delta) if conformal_delta is not None else None
            ),
            ensemble_scale=float(ensemble_scale),
            ensemble_bias=float(ensemble_bias),
            masking_scale=float(masking_scale),
            masking_bias=float(masking_bias),
            has_ensemble=bool(has_ens),
            has_masking=bool(has_msk),
            u0=float(u0),
            u_scale=float(s),
            n_fit=int(len(np.asarray(mu))),
            extra=dict(extra or {}),
        )
        return cls(fit_obj)

    # ----- inference -----

    def apply(
        self,
        *,
        mu: np.ndarray,
        sigma: np.ndarray | None = None,
        q_lo: np.ndarray | None = None,
        q_hi: np.ndarray | None = None,
        disagreement: np.ndarray | None = None,
        masking_std: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply the calibration to a fresh batch.

        Returns a dict with ``pred``, ``lower``, ``upper``, ``uncertainty``
        (fused, in sigma-units), ``confidence_score`` (in [0,1]), and the
        sigma-unit per-source components for downstream diagnostics.
        """
        fp = self.fit_params
        mu = np.asarray(mu, dtype=float)

        if fp.head_kind == "gaussian":
            if sigma is None:
                raise ValueError("gaussian apply requires sigma")
            sigma = np.asarray(sigma, dtype=float)
            T = float(fp.temperature)
            sigma_cal = T * sigma
            half = self.z_alpha * sigma_cal
            lower = mu - half
            upper = mu + half
            primary_sigma = sigma_cal
        else:
            if q_lo is None or q_hi is None:
                raise ValueError("quantile apply requires q_lo and q_hi")
            q_lo = np.asarray(q_lo, dtype=float)
            q_hi = np.asarray(q_hi, dtype=float)
            delta = float(fp.conformal_delta)
            lower = q_lo - delta
            upper = q_hi + delta
            primary_sigma = (upper - lower) / 2.0

        sigma_ens = None
        if disagreement is not None and fp.has_ensemble:
            sigma_ens = apply_monotone_scaling(
                disagreement, fp.ensemble_scale, fp.ensemble_bias
            )
        sigma_msk = None
        if masking_std is not None and fp.has_masking:
            sigma_msk = apply_monotone_scaling(
                masking_std, fp.masking_scale, fp.masking_bias
            )

        u = fuse_uncertainty(primary_sigma, sigma_ens, sigma_msk)
        conf = confidence_from_u(u, fp.u0, fp.u_scale)

        return {
            "pred": mu.astype(np.float32),
            "lower": lower.astype(np.float32),
            "upper": upper.astype(np.float32),
            "uncertainty": u.astype(np.float32),
            "confidence_score": conf.astype(np.float32),
            "sigma_primary": primary_sigma.astype(np.float32),
            "sigma_ensemble": (
                sigma_ens.astype(np.float32) if sigma_ens is not None else None
            ),
            "sigma_masking": (
                sigma_msk.astype(np.float32) if sigma_msk is not None else None
            ),
        }

    # ----- serialisation -----

    def to_dict(self) -> dict:
        return {"version": 1, "fit": asdict(self.fit_params)}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        if d.get("version") != 1:
            raise ValueError(f"unsupported calibrator version: {d.get('version')!r}")
        fit = CalibrationFit(**d["fit"])
        return cls(fit)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        with open(path) as f:
            return cls.from_dict(json.load(f))
