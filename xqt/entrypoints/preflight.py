"""Run XQT recipe preflight checks from environment configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

from xqt.pipeline.preflight import preflight_xqt_config

CONFIG_ENV = "XQT_CONFIG"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "recipes" / "smoke_cpu.yaml"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run preflight without parsing command-line arguments."""

    del argv
    config_path = Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG))).expanduser()
    report = preflight_xqt_config(config_path)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
