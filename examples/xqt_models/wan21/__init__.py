"""Compatibility alias for the migrated Wan 2.1 package."""

import sys
from xqt.model import wan21 as _implementation

sys.modules[__name__] = _implementation
