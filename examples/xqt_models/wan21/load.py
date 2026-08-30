"""Compatibility alias for the migrated Wan 2.1 loader."""

import sys
from xqt.model.wan21 import load as _implementation

sys.modules[__name__] = _implementation
