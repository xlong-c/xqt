"""Compatibility alias for the migrated Flux.2 Klein targets."""

import sys
from xqt.model.flux2_klein import targets as _implementation

sys.modules[__name__] = _implementation
