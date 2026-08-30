"""Compatibility alias for the migrated Flux.2 Klein optimizer."""

import sys
from xqt.model.flux2_klein import optimize as _implementation

sys.modules[__name__] = _implementation
