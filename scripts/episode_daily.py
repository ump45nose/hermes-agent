#!/usr/bin/env python3
"""Daily fail-open Episode backlog processor."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from agent.episode_memory import extract_one_candidate, subject_id
from agent.knowledge_distillation import distill_profile_memory
from hermes_state import SessionDB


def _profile_databases(root: Path):
    profiles = root / "profiles"
    if profiles.is_dir():
        for directory in sorted(profiles.iterdir()):
            if directory.is_dir():
                yield directory.name, directory, directory / "state.db"


@contextmanager
def _active_profile(home: Path):
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


def _distillation_config(home: Path) -> dict:
    try:
        import yaml

        config = yaml.safe_load((home / "config.yaml").read_text()) or {}
        return ((config.get("episode_memory") or {}).get("distillation") or {})
    except Exception:
        return {}


def _parse_backfill_since(value: str) -> float:
    normalized = str(value or "").strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("--backfill-since must include an explicit timezone")
    return parsed.timestamp()


def run(
    *,
    root: Path,
    max_segments: int,
    max_seconds: int,
    backfill_since: float | None = None,
) -> dict:
    started = time.monotonic()
    retry_failed_before = time.time()
    report = {
        "processed": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
        "memory_profiles": 0,
        "memory_episodes": 0,
        "memory_writes": 0,
        "memory_failed": 0,
        "remaining": False,
        "backfill_since": backfill_since,
        "profiles": {},
    }
    states = []
    for profile, home, path in _profile_databases(root):
        if not path.exists():
            continue
        db = SessionDB(db_path=path)
        try:
            with _active_profile(home):
                distillation_config = _distillation_config(home)
                enqueued = db.enqueue_episode_backlog(
                    profile=profile,
                    limit=5_000,
                    since_timestamp=backfill_since,
                )
            profile_report = {
                "enqueued": int(enqueued),
                "processed": 0,
                "succeeded": 0,
                "skipped": 0,
                "failed": 0,
                "backlog": 0,
            }
            report["profiles"][profile] = profile_report
            states.append(
                {
                    "profile": profile,
                    "home": home,
                    "db": db,
                    "distillation": distillation_config,
                    "report": profile_report,
                }
            )
        except Exception as exc:
            db.close()
            report["failed"] += 1
            report["profiles"][profile] = {
                "enqueued": 0,
                "processed": 0,
                "succeeded": 0,
                "skipped": 0,
                "failed": 1,
                "backlog": 0,
                "error_type": type(exc).__name__,
            }
            continue

    try:
        while True:
            progressed = False
            for state in states:
                profile_report = state["report"]
                if profile_report["processed"] >= max_segments:
                    continue
                if time.monotonic() - started >= max_seconds:
                    report["remaining"] = True
                    break
                db = state["db"]
                claimed = None
                sessions = db.list_episode_candidate_sessions(limit=500)
                with _active_profile(state["home"]):
                    for session in sessions:
                        result = extract_one_candidate(
                            db,
                            session_id=str(session["id"]),
                            profile=state["profile"],
                            subject=subject_id(
                                str(session.get("source") or "local"),
                                str(session.get("user_id") or "local-owner"),
                            ),
                            trigger_source="daily_backlog",
                            retry_failed=True,
                            retry_failed_before=retry_failed_before,
                            extractor_model="episode_extraction",
                        )
                        if str(result.get("status") or "none") != "none":
                            claimed = result
                            break
                if claimed is None:
                    continue
                progressed = True
                status = str(claimed.get("status") or "failed")
                profile_report["processed"] += 1
                report["processed"] += 1
                if status in {"succeeded", "skipped", "failed"}:
                    profile_report[status] += 1
                    report[status] += 1
            if report["remaining"] or not progressed:
                break

        for state in states:
            db = state["db"]
            profile_report = state["report"]
            profile_report["backlog"] = db.count_episode_backlog()
            if profile_report["backlog"]:
                report["remaining"] = True
                continue
            if state["distillation"].get("enabled") is not True:
                continue
            with _active_profile(state["home"]):
                memory = distill_profile_memory(
                    db,
                    profile=state["profile"],
                    profile_home=state["home"],
                    owner_subjects=[
                        str(value)
                        for value in (
                            state["distillation"].get("owner_subjects") or []
                        )
                        if str(value).strip()
                    ],
                    limit=50,
                )
            report["memory_profiles"] += 1
            report["memory_episodes"] += int(memory.get("processed") or 0)
            report["memory_writes"] += int(memory.get("writes") or 0)
            if memory.get("status") == "failed":
                report["memory_failed"] += 1
    finally:
        for state in states:
            state["db"].close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-segments", type=int, default=50)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument(
        "--backfill-since",
        default="",
        help=(
            "RFC3339 migration watermark for daily discovery. Complete turns "
            "before this instant are not backfilled; finalized new turns still "
            "enter the queue transactionally."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/hermes/.hermes"),
    )
    args = parser.parse_args()
    try:
        backfill_since = (
            _parse_backfill_since(args.backfill_since)
            if str(args.backfill_since or "").strip()
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    report = run(
        root=args.root,
        max_segments=max(1, args.max_segments),
        max_seconds=max(60, args.max_seconds),
        backfill_since=backfill_since,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if report["failed"] or report["memory_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
