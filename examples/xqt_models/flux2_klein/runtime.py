"""Compatibility alias for the migrated Flux.2 Klein runtime helpers."""

import sys
from xqt.model.flux2_klein import runtime as _implementation

sys.modules[__name__] = _implementation
