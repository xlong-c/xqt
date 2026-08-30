"""Protocols for concrete model-family adapters."""

from __future__ import annotations

from typing import Any


class ModelAdapter:
    """Base class for model-specific loading and compatibility adaptation."""

    def load(self, checkpoint: str | None, **params: Any) -> Any:
        """Load a model from a checkpoint or repository identifier."""

        raise NotImplementedError

    def adapt(self, model: Any, **params: Any) -> Any:
        """Adapt an already loaded model for XQT consumption."""

        return model

    def structure_contract(self, model: Any) -> Any:
        """Optionally return a model structure contract."""

        del model
        return None

    def inference_contract(self, model: Any) -> Any:
        """Optionally return a semantic inference contract."""

        del model
        return None


__all__ = ["ModelAdapter"]
