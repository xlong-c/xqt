"""Run an XQT stage workflow from environment configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from xqt.workflows import optimize_model

CONFIG_ENV = "XQT_WORKFLOW_CONFIG"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "recipes" / "detection" / "yolo_detection_smoke.yaml"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the configured stage workflow without parsing command-line arguments."""

    del argv
    config_path = Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG))).expanduser()
    result = optimize_model(config_path)
    print(f"project: {result.context.config.project.name}")
    print(f"stages: {','.join(stage.name for stage in result.stages)}")
    if result.best_stage is not None:
        print(f"best_stage: {result.best_stage}")
    workflow_manifest = result.context.artifacts.get("workflow_manifest")
    if workflow_manifest is not None:
        print(f"workflow_manifest: {workflow_manifest}")
    workflow_result = result.context.artifacts.get("workflow_result")
    if workflow_result is not None:
        print(f"workflow_result: {workflow_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
