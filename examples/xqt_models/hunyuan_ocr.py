"""Compatibility alias for the migrated XQT HunyuanOCR adapter."""

import sys
from xqt.model import hunyuan_ocr as _implementation

sys.modules[__name__] = _implementation
