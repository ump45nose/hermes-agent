#!/usr/bin/env python3
"""Switch deployed profiles to the canonical failures Tool Context Editor mode."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


DEFAULT_PROFILES = ("lingjun", "companion", "ops", "research", "xp")


def _write_yaml_atomic(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                value,
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate(root: Path, profiles: tuple[str, ...]) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "backups" / "tool-context-editor-failures" / stamp
    backup.mkdir(parents=True, mode=0o700)
    backup.chmod(0o700)

    changes: list[dict[str, str]] = []
    for profile in profiles:
        config_path = root / "profiles" / profile / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing profile config: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError(f"profile config must be a mapping: {config_path}")

        destination = backup / profile / "config.yaml"
        destination.parent.mkdir(mode=0o700)
        shutil.copy2(config_path, destination)
        destination.chmod(0o600)

        editor = config.setdefault("tool_context_editor", {})
        if not isinstance(editor, dict):
            raise ValueError(
                f"tool_context_editor must be a mapping: {config_path}"
            )
        before = str(editor.get("mode") or "")
        editor["mode"] = "failures"
        if before != "failures":
            _write_yaml_atomic(config_path, config)
        changes.append(
            {"profile": profile, "before": before, "after": "failures"}
        )

    return {"backup": str(backup), "changes": changes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/hermes/.hermes"))
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
    )
    args = parser.parse_args()
    result = migrate(args.root.expanduser().resolve(), tuple(args.profiles))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
