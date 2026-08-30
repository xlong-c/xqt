"""Compatibility alias for the migrated Wan 2.1 optimizer."""

import sys
from xqt.model.wan21 import optimize as _implementation

sys.modules[__name__] = _implementation
