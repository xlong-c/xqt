#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.kernels.wrappers.capability import list_operator_engine_capabilities
from xqt.kernels.ops._impl.engines.cutile import list_cutile_kernel_specs
from xqt.kernels.ops._impl.engines.cutlass import list_cutlass_kernel_specs
from xqt.kernels.ops._impl.engines.cute_dsl import list_cute_dsl_kernel_specs
from xqt.kernels.ops._impl.engines.tilelang import list_tilelang_kernel_specs
from xqt.kernels.ops._impl.engines.triton import list_triton_kernel_specs


def _sorted_keys(payload: dict[str, Any]) -> list[str]:
    return sorted(payload.keys())


def main() -> None:
    inventory = {
        "capabilities": list_operator_engine_capabilities(),
        "kernels": {
            "triton": _sorted_keys(list_triton_kernel_specs()),
            "tilelang": _sorted_keys(list_tilelang_kernel_specs()),
            "cutile": _sorted_keys(list_cutile_kernel_specs()),
            "cutlass": _sorted_keys(list_cutlass_kernel_specs()),
            "cute_dsl": _sorted_keys(list_cute_dsl_kernel_specs()),
        },
    }
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
