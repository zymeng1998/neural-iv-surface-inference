"""Model definitions for IV surface inference."""

from neural_iv_surface_inference.models.baseline_mlp import BaselineMLP
from neural_iv_surface_inference.models.interpolation import (
    run_interpolation_baseline,
)
from neural_iv_surface_inference.models.losses import (
    MaskedMSELoss,
    MaskedMAELoss,
    CombinedLoss,
)
