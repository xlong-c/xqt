"""Shared runtime types for XQT passes."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .artifact import ArtifactManifest
from .schema import XQTConfig


@dataclass
class XQTContext:
    """Mutable context passed between XQT pipeline passes."""

    config: XQTConfig
    model: Any = None
    teacher: Any = None
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    manifest: Optional[ArtifactManifest] = None

    def require_model(self) -> Any:
        """Return the current model or raise a clear error."""

        if self.model is None:
            raise ValueError("XQTContext.model is required")
        return self.model


__all__ = ["XQTContext"]
