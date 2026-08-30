"""Compatibility alias for the migrated Flux.2 Klein loader."""

import sys
from xqt.model.flux2_klein import load as _implementation

sys.modules[__name__] = _implementation
