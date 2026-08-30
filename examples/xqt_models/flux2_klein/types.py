"""Compatibility alias for the migrated Flux.2 Klein types."""

import sys
from xqt.model.flux2_klein import types as _implementation

sys.modules[__name__] = _implementation
