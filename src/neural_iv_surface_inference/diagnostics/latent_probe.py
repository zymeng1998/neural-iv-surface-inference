"""Latent extraction for the conditional surface model (Story 2E.2).

Single helper: register a forward hook on
:attr:`~neural_iv_surface_inference.models.conditional_surface.ConditionalSurfaceModel.encoder`,
run the model in ``eval()`` mode under ``torch.no_grad()`` over the supplied
loader, and return the stacked per-date latent matrix ``Z`` of shape
``[N_dates, latent_dim]``.

The encoder's ``forward`` already returns ``z_t`` (it is the output of the
post-pool MLP — see :func:`SetEncoder.forward`). Hooking the encoder module
output rather than its inner ``post_mlp`` is therefore equivalent and one
less brittle attribute path.

Per-batch payloads (query coordinates, target IV, query masks) are returned
alongside ``Z`` so the caller can feed
:mod:`~neural_iv_surface_inference.diagnostics.contribution` ablation
utilities without re-iterating the loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatentCache:
    """Captured latents and the per-row payload needed for decoder evaluation.

    All tensors are CPU-resident and detached. Row index ``i`` of ``Z`` aligns
    with row ``i`` of every payload tensor.

    Attributes
    ----------
    Z
        ``[N, latent_dim]`` — stacked encoder outputs across all batches.
    query
        ``[N, Q, coord_dim]`` — query coordinates for each row.
    target
        ``[N, Q]`` — target IV for each row.
    query_mask
        ``[N, Q]`` bool — True for real query points, False for padding.
    """

    Z: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor
    query_mask: torch.Tensor


def extract_latents(
    model: nn.Module,
    loader: Iterable[dict],
    device: torch.device,
) -> LatentCache:
    """Extract per-date latents and per-row payloads in a single pass.

    Parameters
    ----------
    model
        A ``ConditionalSurfaceModel`` (or any module with an ``encoder``
        attribute whose forward returns ``[B, latent_dim]``).
    loader
        Iterable of batch dicts with keys
        ``{"context", "context_mask", "query", "target", "query_mask"}`` —
        the contract used by ``train_conditional()``.
    device
        Device to move each batch to before forwarding.

    Returns
    -------
    LatentCache
        All tensors on CPU; ``Z`` has shape ``[N, latent_dim]``.

    Raises
    ------
    AttributeError
        If ``model`` has no ``encoder`` attribute to hook.
    RuntimeError
        If the loader yields no batches.
    """
    if not hasattr(model, "encoder"):
        raise AttributeError(
            "model has no 'encoder' attribute; cannot register a latent hook"
        )

    model.eval()
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        captured.append(output.detach().to("cpu"))

    handle = model.encoder.register_forward_hook(hook)
    queries: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    try:
        with torch.no_grad():
            for raw in loader:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in raw.items()
                }
                _ = model(batch["context"], batch["context_mask"], batch["query"])
                queries.append(batch["query"].detach().to("cpu"))
                targets.append(batch["target"].detach().to("cpu"))
                masks.append(batch["query_mask"].detach().to("cpu"))
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("loader yielded no batches; latent cache is empty")

    Z = torch.cat(captured, dim=0)
    query = torch.cat(queries, dim=0)
    target = torch.cat(targets, dim=0)
    query_mask = torch.cat(masks, dim=0)

    if not (Z.shape[0] == query.shape[0] == target.shape[0] == query_mask.shape[0]):
        raise RuntimeError(
            "row-count mismatch between Z and payload tensors: "
            f"{Z.shape[0]}, {query.shape[0]}, {target.shape[0]}, {query_mask.shape[0]}"
        )

    _LOG.info(
        "extracted latents: Z shape %s, query shape %s",
        tuple(Z.shape), tuple(query.shape),
    )
    return LatentCache(Z=Z, query=query, target=target, query_mask=query_mask)


__all__ = ["LatentCache", "extract_latents"]
