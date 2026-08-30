"""Compatibility alias for the migrated XQT Unlimited-OCR adapter."""

import sys
from xqt.model import unlimited_ocr as _implementation

sys.modules[__name__] = _implementation
