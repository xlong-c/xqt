"""Compatibility alias for the migrated Wan 2.1 types."""

import sys
from xqt.model.wan21 import types as _implementation

sys.modules[__name__] = _implementation
