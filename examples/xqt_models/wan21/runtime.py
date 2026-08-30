"""Compatibility alias for the migrated Wan 2.1 runtime helpers."""

import sys
from xqt.model.wan21 import runtime as _implementation

sys.modules[__name__] = _implementation
