"""XQT semantic Linear / Conv2d / LayerNorm facades."""

from __future__ import annotations

from typing import Any

from torch import nn

from ._mixin import _SemanticModuleMixin


class Linear(nn.Linear, _SemanticModuleMixin):
    """XQT semantic Linear facade with explicit runtime intent."""

    def __init__(self, *args: Any, engine: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_runtime_intent(engine=engine)


class Conv2d(nn.Conv2d, _SemanticModuleMixin):
    """XQT semantic Conv2d facade with explicit runtime intent."""

    def __init__(self, *args: Any, engine: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_runtime_intent(engine=engine)


class LayerNorm(nn.LayerNorm, _SemanticModuleMixin):
    """XQT semantic LayerNorm facade with explicit runtime intent."""

    def __init__(self, *args: Any, engine: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_runtime_intent(engine=engine)
