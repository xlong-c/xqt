"""Sequential pass manager for XQT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from xqt.core.errors import XQTPipelineError
from xqt.core.types import XQTContext


class XQTPass(Protocol):
    """Protocol implemented by XQT passes."""

    name: str

    def run(self, context: XQTContext) -> XQTContext:
        """Run the pass and return the updated context."""


@dataclass
class SequentialPipeline:
    """Run XQT passes in order."""

    passes: List[XQTPass]

    @classmethod
    def from_iterable(cls, passes: Iterable[XQTPass]) -> "SequentialPipeline":
        """Build a pipeline from any pass iterable."""

        return cls(list(passes))

    def run(self, context: XQTContext) -> XQTContext:
        """Run all passes and record pass names in the manifest when present."""

        current = context
        for pipeline_pass in self.passes:
            try:
                current = pipeline_pass.run(current)
            except Exception as exc:
                raise XQTPipelineError(f"Pass '{pipeline_pass.name}' failed: {exc}") from exc

            if current.manifest is not None:
                current.manifest.passes.append(pipeline_pass.name)
        return current


__all__ = ["SequentialPipeline", "XQTPass"]
