"""Minimal smoke tests for project imports and basic sanity."""


def test_package_imports():
    """Verify that the main package is importable."""
    import neural_iv_surface_inference  # noqa: F401


def test_torch_available():
    """Verify that PyTorch is importable."""
    import torch
    assert hasattr(torch, "__version__")
