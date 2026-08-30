"""Inventory of xqt.kernels registry."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xqt.kernels  # noqa: F401
import xqt.kernels.ops  # noqa: F401  # trigger group registration
from xqt.kernels.registry import registry


def main() -> None:
    for op in registry.ops():
        specs = registry.get(op)
        print(f"{op}:")
        for s in specs:
            caps = sorted(c.device.value for c in s.capabilities) if s.capabilities else ["any"]
            print(f"  - {s.backend.value:12} target={s.target} caps={caps} desc={s.description!r}")
    print(f"\nTotal ops: {len(registry.ops())}, total specs: {len(registry.all_specs())}")


if __name__ == "__main__":
    main()
