#!/usr/bin/env python3
"""One-time, reversible migration for fixed profile prompts and local context."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from agent.local_context import LocalContextStore
from hermes_cli.prompt_compiler import (
    ensure_profile_governance_config,
    write_compiled_prompt,
)


PROFILES = ("default", "lingjun", "companion", "ops", "research", "xp")
MEM0_LIST_URL = "https://api.mem0.ai/v3/memories/"


def _profile_home(root: Path, name: str) -> Path:
    return root if name == "default" else root / "profiles" / name


def _backup(root: Path, homes: dict[str, Path]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / "backups" / "prompt-context-governance" / stamp
    target.mkdir(parents=True, exist_ok=False)
    target.chmod(0o700)
    for name, home in homes.items():
        out = target / name
        out.mkdir(mode=0o700)
        for relative in (
            "config.yaml",
            "SOUL.md",
            "AGENTS.md",
            "memories/MEMORY.md",
            "memories/USER.md",
            "prompt/system.md",
            "prompt/prompt.lock.yaml",
        ):
            source = home / relative
            if source.exists() or source.is_symlink():
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_symlink():
                    destination.write_text(
                        f"SYMLINK -> {os.readlink(source)}\n", encoding="utf-8"
                    )
                elif source.is_file():
                    shutil.copy2(source, destination)
                try:
                    destination.chmod(0o600)
                except OSError:
                    pass
    return target


def _set_config(name: str, home: Path) -> None:
    ensure_profile_governance_config(home)
    path = home / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config.setdefault("tool_context_editor", {})["mode"] = "report_only"
    config.setdefault("observability", {}).setdefault("request_snapshot", "off")
    config.setdefault("episode_store", {}).update(
        {
            "backend": "local",
            "database": str(home.parent.parent / "user-context" / "context.db")
            if home.parent.name == "profiles"
            else str(home / "user-context" / "context.db"),
            "mem0_shadow_until": (
                datetime.now(timezone.utc) + timedelta(days=7)
            ).isoformat(),
            "delete_remote": False,
        }
    )
    tools = config.setdefault("tools", {})
    tools.setdefault("kanban", {})["worker_only"] = name in {
        "ops", "research", "xp", "companion"
    }
    platform = config.setdefault("platform_toolsets", {})
    if name == "research":
        for surface in ("cli", "cron"):
            exposure = platform.setdefault(surface, {"direct": [], "deferred": []})
            deferred = exposure.setdefault("deferred", [])
            if "delegation" not in deferred:
                deferred.append("delegation")
        platform["subagent"] = {
            "direct": ["tool_artifact"],
            "deferred": ["web", "browser", "context7", "smart-search"],
        }
        config.setdefault("delegation", {})["research_leaf_toolsets"] = [
            "web", "browser", "context7", "smart-search", "tool_artifact"
        ]
    if name == "xp":
        for exposure in platform.values():
            if not isinstance(exposure, dict):
                continue
            for group in ("direct", "deferred"):
                if isinstance(exposure.get(group), list):
                    exposure[group] = [
                        item for item in exposure[group] if item != "delegation"
                    ]
        cli = platform.setdefault("cli", {"direct": [], "deferred": []})
        deferred = cli.setdefault("deferred", [])
        for capability in ("web", "browser", "smart-search", "session_search"):
            if capability not in deferred:
                deferred.append(capability)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _clean_soul(name: str, home: Path) -> str:
    path = home / "SOUL.md"
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    if name == "lingjun":
        marker = "张灵珺:"
        new = old[old.find(marker):] if marker in old else old
    elif name == "companion":
        new = old
    elif name == "ops":
        new = "表达风格：直接、克制、可审计；技术结论先给证据和结果。\n"
    elif name == "research":
        new = "表达风格：严谨、克制、先证据后结论；明确标注不确定性。\n"
    elif name == "xp":
        new = "表达风格：事务性、中立、就事论事，不跨越用户指定的资源范围。\n"
    else:
        new = old
    path.write_text(new.rstrip() + "\n", encoding="utf-8")
    path.chmod(0o600)
    return "reduced" if new != old else "preserved"


def _clean_memory(home: Path) -> dict[str, Any]:
    path = home / "memories" / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.write_text("", encoding="utf-8")
    path.chmod(0o600)
    return {
        "bytes_before": len(old.encode()),
        "bytes_after": 0,
        "classification": (
            "moved_to_compiled_prompt_or_regression_tests"
            if old else "already_empty"
        ),
    }


def _remove_agents_link(home: Path) -> bool:
    path = home / "AGENTS.md"
    if not path.is_symlink():
        return False
    path.unlink()
    return True


def _mem0_request(
    key: str,
    payload: dict[str, Any],
    *,
    page: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{MEM0_LIST_URL}?page={page}&page_size=100",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Token {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode())
    return result if isinstance(result, dict) else {}


def _migrate_remote_episodes(root: Path, homes: dict[str, Path]) -> dict[str, int]:
    key = os.environ.get("MEM0_EPISODES_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MEM0_EPISODES_API_KEY is not available in this process")
    subjects: set[str] = set()
    for base in [root, *homes.values()]:
        directory = base / "episode-sync" / "users"
        if directory.is_dir():
            subjects.update(path.name for path in directory.iterdir() if path.is_dir())
    store = LocalContextStore(root / "user-context" / "context.db")
    imported = seen = 0
    for subject in sorted(subjects):
        for page in range(1, 6):
            response = _mem0_request(
                key,
                {
                    "filters": {
                        "AND": [{"user_id": f"hermes-episodes:{subject}"}]
                    }
                },
                page=page,
            )
            results = response.get("results") or []
            for item in results:
                metadata = item.get("metadata") or {}
                if metadata.get("memory_kind") != "episode":
                    continue
                seen += 1
                body = item.get("memory")
                if not isinstance(body, str):
                    body = json.dumps(body, ensure_ascii=False, sort_keys=True)
                source_hash = str(metadata.get("source_hash") or "")
                run_id = str(item.get("run_id") or metadata.get("run_id") or item.get("id"))
                if not source_hash:
                    continue
                if store.import_episode(
                    subject_id=subject,
                    profile=str(metadata.get("profile") or "unknown"),
                    run_id=run_id,
                    source_hash=source_hash,
                    body=body,
                    metadata=metadata,
                ):
                    imported += 1
            if len(results) < 100:
                break
    return {"seen": seen, "imported": imported}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/hermes/.hermes"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--migrate-mem0", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    homes = {
        name: _profile_home(root, name)
        for name in PROFILES
        if _profile_home(root, name).is_dir()
    }
    report: dict[str, Any] = {
        "root": str(root),
        "profiles": list(homes),
        "apply": args.apply,
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    backup = _backup(root, homes)
    profile_report: dict[str, Any] = {}
    for name, home in homes.items():
        _set_config(name, home)
        write_compiled_prompt(home, preset=name, model_family="generic")
        profile_report[name] = {
            "agents_link_removed": _remove_agents_link(home),
            "soul": _clean_soul(name, home),
            "memory": _clean_memory(home) if name != "default" else {"preserved": True},
        }
    report["backup"] = str(backup)
    report["profile_report"] = profile_report
    if args.migrate_mem0:
        report["mem0"] = _migrate_remote_episodes(root, homes)
    report_path = backup / "migration-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.chmod(0o600)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
