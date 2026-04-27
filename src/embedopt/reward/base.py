"""Base reward contract for inference-time diffusion steering.

This module exposes the :class:`Reward` abstract base class that downstream
reward terms should inherit from. Rewards encapsulate the logic required to
score intermediate diffusion samples and feed that signal back into the
steering mechanism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, MutableMapping, Optional


StateDict = MutableMapping[str, Any]


class Reward(ABC):
    """Abstract base class for reward terms used during steering.

    Args:
        name: Optional identifier for logging or aggregation. Defaults to the
            class name.

    Subclasses are expected to implement :meth:`compute`, which returns a
    scalar-compatible value (e.g. ``float`` or ``torch.Tensor`` with a single
    element). The base implementation of :meth:`__call__` simply forwards to
    :meth:`compute`.

    Subclassing guide:
        * Define a custom ``__init__`` that accepts the context or configuration
          required for the reward term, store it on ``self``, and call
          ``super().__init__(name=...)`` so the base metadata is registered.
        * Implement :meth:`compute` to read from ``self`` and produce the reward
          value for each sample.
        * Override :meth:`reset`, :meth:`state_dict`, and :meth:`load_state_dict`
          when the reward maintains mutable state across calls.
    """

    def __init__(self, name: Optional[str] = None, agg: Optional[str] = None) -> None:
        """Store shared metadata for reward instances.

        Args:
            name: Optional identifier for logging or aggregation.
            agg: Aggregation mode for multi-sample rewards. ``"mean"`` averages
                per-sample rewards (embedopt), ``"sum"`` sums them (dps),
                ``None`` returns the raw ``(N_sample,)`` tensor.
        """
        if agg is not None and agg not in ("mean", "sum"):
            raise ValueError(f"agg must be 'mean', 'sum', or None, got '{agg}'")

        self._name = name or self.__class__.__name__
        self.agg = agg

    @property
    def name(self) -> str:
        """Human-readable identifier for the reward instance."""

        return self._name

    def __call__(self, sample: Any, pdb_save_path: Optional[str] = None) -> Any:
        """Evaluate the reward for the given sample."""

        return self.compute(sample, pdb_save_path)

    @abstractmethod
    def compute(self, sample: Any, pdb_save_path: Optional[str] = None) -> Any:
        """Compute the reward value for ``sample``."""

    def reset(self) -> None:
        """Reset any internal state (no-op by default)."""

    def state_dict(self) -> StateDict:
        """Serialize the reward's mutable state.

        Returns:
            A mapping representing the internal state. The default
            implementation returns an empty dict.
        """

        return {}

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore the reward's state from ``state_dict``."""

        if state_dict:
            raise ValueError(f"{self.__class__.__name__} received unexpected state entries: " f"{', '.join(state_dict.keys())}")
