"""Compatibility alias for the migrated XQT Wan 2.1 adapter."""

import sys
from xqt.model import wan21_vae as _implementation

sys.modules[__name__] = _implementation
