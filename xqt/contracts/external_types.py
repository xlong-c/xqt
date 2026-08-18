"""External quant info value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExternalQuantInfo:
    """Parsed external quantization metadata for one model directory."""

    format: str
    source_file: str
    raw_config: dict[str, Any] = field(default_factory=dict)
    bits: int | None = None
    group_size: int | None = None
    sym: bool | None = None
    desc_act: bool | None = None
    zero_point: bool | None = None
    quant_method_raw: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    source_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        sources = self.source_files or ((self.source_file,) if self.source_file else ())
        return {
            "format": self.format,
            "source_file": self.source_file,
            "source_files": list(sources),
            "raw_config": dict(self.raw_config),
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "desc_act": self.desc_act,
            "zero_point": self.zero_point,
            "quant_method_raw": self.quant_method_raw,
            "extra": dict(self.extra),
        }


__all__ = ["ExternalQuantInfo"]
