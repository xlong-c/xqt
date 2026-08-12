"""Compatibility re-export for the in-progress arch-subpackage layout.

The ``xqt.gemm.backends.sm89`` subpackage was copied from the flat backend
modules and expects sibling modules at ``xqt.gemm.backends`` level.  This
shim keeps those verbatim copies importable by re-exporting the canonical
flat implementation; remove together with the flat re-export when the
layout migration lands.
"""

from .. import fp8 as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("_")})
__all__ = [name for name in dir(_impl) if not name.startswith("_")]
