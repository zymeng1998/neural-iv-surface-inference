"""Phase 2 W2 diagnostics subpackage.

Sensitivity- and structure-based reliability signals computed on top of the
model-agnostic ``Predictor`` interface (W1, story 2A.2). Currently exposes the
masking-sensitivity harness (story 2B.2), the no-arbitrage diagnostics
(story 2B.3), and the risk-flag synthesis + region binning (story 2B.4); the
diagnostics runner (2B.5) lands later.
"""

from __future__ import annotations

from neural_iv_surface_inference.diagnostics.masking_sensitivity import (
    MaskingSensitivityResult,
    instability_summary,
    mask_resample,
    masking_sensitivity,
)
from neural_iv_surface_inference.diagnostics.no_arbitrage import (
    ViolationResult,
    calendar_violations,
    convexity_violations,
    monotonicity_violations,
    no_arb_diagnostics,
)
from neural_iv_surface_inference.diagnostics.risk_flags import (
    RiskFlagConfig,
    RiskFlagResult,
    bin_to_regions,
    derive_risk_flags,
)

__all__ = [
    "MaskingSensitivityResult",
    "instability_summary",
    "mask_resample",
    "masking_sensitivity",
    "ViolationResult",
    "calendar_violations",
    "convexity_violations",
    "monotonicity_violations",
    "no_arb_diagnostics",
    "RiskFlagConfig",
    "RiskFlagResult",
    "bin_to_regions",
    "derive_risk_flags",
]
