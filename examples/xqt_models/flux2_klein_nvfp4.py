"""Compatibility alias for the migrated XQT Flux.2 Klein adapter."""

import sys
from xqt.model import flux2_klein_nvfp4 as _implementation

sys.modules[__name__] = _implementation
