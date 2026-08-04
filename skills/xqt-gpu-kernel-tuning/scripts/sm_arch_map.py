#!/usr/bin/env python3
from __future__ import annotations

import json


SM_ARCH_MAP = {
    "sm_80": {"family": "Ampere", "compute_capability": "8.0", "typical_gpus": ["A100", "A30"]},
    "sm_86": {"family": "Ampere", "compute_capability": "8.6", "typical_gpus": ["A40", "RTX A6000", "RTX 3090"]},
    "sm_89": {"family": "Ada", "compute_capability": "8.9", "typical_gpus": ["L40S", "RTX 4090", "RTX 6000 Ada"]},
    "sm_90": {"family": "Hopper", "compute_capability": "9.0", "typical_gpus": ["H100", "H200", "GH200"]},
    "sm_100": {"family": "Blackwell", "compute_capability": "10.0", "typical_gpus": ["B200", "GB200"]},
    "sm_103": {"family": "Blackwell", "compute_capability": "10.3", "typical_gpus": ["B300", "GB300"]},
    "sm_120": {"family": "Blackwell", "compute_capability": "12.0", "typical_gpus": ["RTX 5090", "RTX PRO 6000 Blackwell"]},
    "sm_121": {"family": "Blackwell", "compute_capability": "12.1", "typical_gpus": ["GB10"]},
}


def main() -> None:
    print(json.dumps(SM_ARCH_MAP, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
