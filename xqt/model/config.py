"""Model compatibility profiles used by concrete XQT model adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelProfile:
    """Serializable selection record for one concrete model adapter."""

    profile_id: str
    family: str = "unknown"
    loader_target: str | None = None
    loader_params: Mapping[str, Any] = field(default_factory=dict)
    adapter_target: str | None = None
    structure_contract: str | None = None
    inference_adapter: str | None = None
    requirements: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("ModelProfile.profile_id must be a non-empty string")
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("ModelProfile.family must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable profile mapping."""

        return asdict(self)


__all__ = ["ModelProfile"]
