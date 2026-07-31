#!/usr/bin/env python3
"""Three-day Profile-local Skill/Tool candidate consolidation."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

from agent.knowledge_distillation import consolidate_profile_candidates
from hermes_state import SessionDB


def _enabled(home: Path) -> bool:
    try:
        import yaml

        config = yaml.safe_load((home / "config.yaml").read_text()) or {}
        return (
            ((config.get("episode_memory") or {}).get("distillation") or {}).get(
                "enabled"
            )
            is True
        )
    except Exception:
        return False


@contextmanager
def _profile(home: Path):
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(home)
    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )

    secret_token = set_secret_scope(build_profile_secret_scope(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous


def run(root: Path, limit: int) -> dict:
    report = {"profiles": 0, "processed": 0, "updates": 0, "failed": 0}
    ready: list[dict] = []
    for home in sorted((root / "profiles").iterdir()):
        path = home / "state.db"
        if not home.is_dir() or not path.exists() or not _enabled(home):
            continue
        db = SessionDB(db_path=path)
        try:
            with _profile(home):
                result = consolidate_profile_candidates(
                    db, profile=home.name, limit=limit
                )
            report["profiles"] += 1
            report["processed"] += int(result.get("processed") or 0)
            report["updates"] += int(result.get("updates") or 0)
            if result.get("status") == "failed":
                report["failed"] += 1
            rows = db.list_learning_candidates(
                statuses=("ready_for_review",), limit=50
            )
            for row in rows:
                if row.get("last_notified_version") == row.get("version_hash"):
                    continue
                payload = json.loads(row.get("payload_json") or "{}")
                ready.append(
                    {
                        "profile": home.name,
                        "candidate_id": int(row["id"]),
                        "kind": row["kind"],
                        "version": row["version_hash"],
                        "title": row["title"],
                        "evidence": (
                            row["successful_run_count"]
                            if row["kind"] == "skill"
                            else max(
                                row["repeated_failure_count"],
                                row["successful_run_count"],
                            )
                        ),
                        "summary": str(
                            payload.get("symptom")
                            or payload.get("trigger")
                            or payload.get("description")
                            or ""
                        )[:240],
                    }
                )
            # The scheduler owns delivery and does not provide an atomic
            # callback into this process. Do not mark a version as notified
            # before Telegram has actually accepted it; an undelivered review
            # must remain eligible for the next three-day digest.
        finally:
            db.close()
    report["ready"] = ready
    return report


def render(report: dict) -> str:
    lines = [
        "Hermes Skill/Tool 学习候选审核",
        (
            f"扫描 {report['profiles']} 个 Profile，处理 {report['processed']} 条 "
            f"Episode，更新 {report['updates']} 个候选。"
        ),
    ]
    if report["failed"]:
        lines.append(f"失败 Profile：{report['failed']}（已保留待下次重试）")
    if not report["ready"]:
        lines.append("本轮没有新的 ready_for_review 候选。")
        return "\n".join(lines)
    for item in report["ready"]:
        lines.extend(
            [
                "",
                (
                    f"[{item['kind']}] {item['profile']}:{item['candidate_id']} "
                    f"v={item['version']} 证据={item['evidence']}"
                ),
                item["title"],
                item["summary"],
            ]
        )
    lines.extend(
        [
            "",
            "回复示例：",
            "批准 skill <profile>:<candidate_id> <version>",
            "拒绝 tool <profile>:<candidate_id> <version>",
            "批准只进入前台修改流程，cron 不会直接修改 Skill 或 Tool。",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/hermes/.hermes"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    report = run(args.root, max(1, args.limit))
    print(render(report))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
