"""Compatibility alias for the migrated Flux.2 Klein package."""

import sys
from xqt.model import flux2_klein as _implementation

sys.modules[__name__] = _implementation
