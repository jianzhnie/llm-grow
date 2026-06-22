"""Pluggable noise strategies for breaking symmetry in expert copies.

By default all expert-cloning expanders use :class:`GaussianNoise`,
which matches the original ExpertClone / Sparse Upcycling papers.
Swap in :class:`UniformNoise` or a custom subclass for experimentation
without forking the expander.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class NoiseStrategy(ABC):
    """Abstract noise-injection strategy for expert duplication.

    Subclasses override :meth:`apply` to implement a specific noise
    distribution.  Instances are passed to expander configs via the
    ``noise`` keyword argument.
    """

    @abstractmethod
    def apply(self, tensor: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
        """Return *tensor* with noise added in place (or as a new tensor).

        Args:
            tensor: The weight tensor to perturb.
            scale:  Noise magnitude (interpretation is strategy-specific).

        Returns:
            A tensor of the same shape and dtype as *tensor*.
        """
        ...


class GaussianNoise(NoiseStrategy):
    """Add zero-mean Gaussian noise: ``tensor += N(0, scale)``."""

    def apply(self, tensor: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * scale


class UniformNoise(NoiseStrategy):
    """Add uniform noise: ``tensor += U(-scale, scale)``."""

    def apply(self, tensor: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
        return tensor + (torch.rand_like(tensor) * 2 - 1) * scale


class ScaledGaussianNoise(NoiseStrategy):
    """Scale-aware Gaussian: ``noise_std = scale * tensor.std()``.

    This matches the noise semantics used in the safetensor-layer
    ``dup_rows_noise_scale`` parameter (noise is relative to tensor std).
    """

    def apply(self, tensor: torch.Tensor, scale: float = 1e-6) -> torch.Tensor:
        noise = torch.randn_like(tensor) * scale * tensor.float().std()
        return tensor + noise.to(tensor.dtype)


# Default instance — used when no explicit strategy is specified.
DEFAULT_NOISE: NoiseStrategy = GaussianNoise()
