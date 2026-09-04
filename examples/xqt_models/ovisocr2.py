"""Compatibility alias for the XQT OvisOCR2 adapter."""

import sys

from xqt.model import ovisocr2 as _implementation

sys.modules[__name__] = _implementation
