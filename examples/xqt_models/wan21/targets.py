"""Compatibility alias for the migrated Wan 2.1 targets."""

import sys
from xqt.model.wan21 import targets as _implementation

sys.modules[__name__] = _implementation
