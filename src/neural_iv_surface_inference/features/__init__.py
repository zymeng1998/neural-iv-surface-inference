"""Feature engineering for IV surface inputs."""

from neural_iv_surface_inference.features.coord_encoding import (
    DEFAULT_INCLUDE_INPUT,
    DEFAULT_MAX_FREQ,
    DEFAULT_NUM_BANDS,
    FourierCoordEncoding,
    RawCoordEncoding,
    build_coord_encoding,
)

__all__ = [
    "DEFAULT_INCLUDE_INPUT",
    "DEFAULT_MAX_FREQ",
    "DEFAULT_NUM_BANDS",
    "FourierCoordEncoding",
    "RawCoordEncoding",
    "build_coord_encoding",
]
