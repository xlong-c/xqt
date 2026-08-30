"""Compatibility alias for the migrated XQT HunyuanOCR TileLang adapter."""

import sys
from xqt.model import hunyuan_ocr_tilelang as _implementation

sys.modules[__name__] = _implementation
