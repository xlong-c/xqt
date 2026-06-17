"""Run an XQT recipe from environment configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from xqt.pipeline.runner import run_xqt_recipe

CONFIG_ENV = "XQT_CONFIG"
WRITE_MANIFEST_ENV = "XQT_WRITE_MANIFEST"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "recipes" / "smoke_cpu.yaml"


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the configured recipe without parsing command-line arguments."""

    del argv
    config_path = Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG))).expanduser()
    context = run_xqt_recipe(
        config_path,
        write_manifest=_env_flag(WRITE_MANIFEST_ENV, default=True),
    )
    manifest = context.artifacts.get("manifest")
    if manifest is not None:
        print(f"manifest: {manifest}")
    print(f"project: {context.config.project.name}")
    print(f"passes: {','.join(context.manifest.passes if context.manifest else [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
