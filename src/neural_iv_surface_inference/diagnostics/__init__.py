"""Diagnostics subpackage.

Phase 2 W2 exports (sensitivity- and structure-based reliability signals on
top of the W1 ``Predictor`` interface): masking-sensitivity (2B.2), no-
arbitrage (2B.3), risk-flag synthesis + region binning (2B.4).

Phase 2 W6 exports (latent-capacity diagnostics for the conditional surface
model, story 2E.2): SVD spectrum (:mod:`effective_rank`), per-dim / per-PC
ablation utilities (:mod:`contribution`), and the encoder forward-hook
latent extractor (:mod:`latent_probe`).
"""

from __future__ import annotations

from neural_iv_surface_inference.diagnostics.contribution import (
    LossFn,
    ablate_dim,
    ablate_pc,
    baseline_loss,
    project_to_pc_basis,
    reconstruct_from_pc_basis,
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
    "RankReport",
    "analyze",
    "LossFn",
    "ablate_dim",
    "ablate_pc",
    "baseline_loss",
    "project_to_pc_basis",
    "reconstruct_from_pc_basis",
    "topk_pc_reconstruction",
    "LatentCache",
    "extract_latents",
]
