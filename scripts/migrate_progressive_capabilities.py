#!/usr/bin/env python3
"""Remove legacy capability-disclosure payloads from one Hermes profile.

The migration is intentionally profile-scoped. Stop that profile's writers
before ``--apply``. A SQLite online backup, config copy, checksums, and manifest
are created before the single migration transaction begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


TRANSIENT_DIRECT = {
    "tool_search",
    "tool_describe",
    "skill_search",
    "skills_list",
    "skill_view",
}
TRANSIENT_UNDERLYING = {"skills_list", "skill_view"}
SKILL_PROMPT_MARKERS = ("<available_skills>", "## Skills (mandatory)")
SKILL_PREFIX = "[IMPORTANT: The user has invoked the "


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_parts(call: Any) -> Tuple[str, str, Any]:
    if not isinstance(call, dict):
        return "", "", None
    call_id = str(call.get("id") or call.get("call_id") or "")
    fn = call.get("function")
    if isinstance(fn, dict):
        return call_id, str(fn.get("name") or ""), fn.get("arguments")
    return call_id, str(call.get("name") or ""), call.get("arguments")


def _args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _is_transient_call(name: str, raw_args: Any) -> bool:
    if name in TRANSIENT_DIRECT:
        return True
    return (
        name == "tool_call"
        and str(_args(raw_args).get("name") or "") in TRANSIENT_UNDERLYING
    )


def _resolved_call(call: dict) -> dict:
    call_id, name, raw_args = _call_parts(call)
    if name != "tool_call":
        return call
    wrapper = _args(raw_args)
    underlying = str(wrapper.get("name") or "")
    if not underlying or underlying in TRANSIENT_UNDERLYING:
        return call
    result = dict(call)
    fn = dict(result.get("function") or {})
    fn["name"] = underlying
    arguments = wrapper.get("arguments", {})
    fn["arguments"] = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    )
    result["function"] = fn
    result["resolved_via"] = "tool_call"
    if call_id and "id" not in result:
        result["id"] = call_id
    return result


def _clean_skill_user_content(content: Any) -> Any:
    if not isinstance(content, str) or not content.startswith(SKILL_PREFIX):
        return content
    from agent.skill_commands import extract_user_instruction_from_skill_message

    extracted = extract_user_instruction_from_skill_message(content)
    if extracted:
        return extracted
    first_line = content.splitlines()[0]
    match = re.search(r'invoked the "([^"]+)"', first_line)
    return f"/{match.group(1).strip()}" if match else "/skill"


def analyze(conn: sqlite3.Connection) -> Dict[str, Any]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, session_id, role, content, tool_call_id, tool_calls, "
        "tool_name, api_content FROM messages ORDER BY id"
    ).fetchall()
    transient_ids: set[str] = set()
    assistant_delete: set[int] = set()
    assistant_updates: Dict[int, str] = {}
    resolved_updates = 0
    affected_sessions: set[str] = set()

    for row in rows:
        raw = row["tool_calls"]
        if not raw:
            continue
        try:
            calls = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(calls, list):
            continue
        kept = []
        changed = False
        for call in calls:
            call_id, name, arguments = _call_parts(call)
            if _is_transient_call(name, arguments):
                changed = True
                if call_id:
                    transient_ids.add(call_id)
                continue
            resolved = _resolved_call(call)
            if resolved != call:
                changed = True
                resolved_updates += 1
            kept.append(resolved)
        if not changed:
            continue
        affected_sessions.add(row["session_id"])
        if not kept and not str(row["content"] or "").strip():
            assistant_delete.add(row["id"])
        else:
            assistant_updates[row["id"]] = (
                json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
                if kept
                else None
            )

    tool_delete: set[int] = set()
    user_updates: Dict[int, Tuple[Any, Any]] = {}
    for row in rows:
        if row["role"] == "tool" and (
            str(row["tool_call_id"] or "") in transient_ids
            or str(row["tool_name"] or "") in TRANSIENT_DIRECT
        ):
            tool_delete.add(row["id"])
            affected_sessions.add(row["session_id"])
        if row["role"] == "user":
            clean_content = _clean_skill_user_content(row["content"])
            clean_api = _clean_skill_user_content(row["api_content"])
            if clean_content != row["content"] or clean_api != row["api_content"]:
                user_updates[row["id"]] = (clean_content, clean_api)
                affected_sessions.add(row["session_id"])

    stale_prompt_sessions = {
        row[0]
        for row in conn.execute(
            "SELECT id FROM sessions WHERE "
            + " OR ".join("system_prompt LIKE ?" for _ in SKILL_PROMPT_MARKERS),
            tuple(f"%{marker}%" for marker in SKILL_PROMPT_MARKERS),
        )
    }
    affected_sessions.update(stale_prompt_sessions)
    return {
        "assistant_delete_ids": assistant_delete,
        "assistant_updates": assistant_updates,
        "tool_delete_ids": tool_delete,
        "user_updates": user_updates,
        "stale_prompt_sessions": stale_prompt_sessions,
        "affected_sessions": affected_sessions,
        "resolved_tool_calls": resolved_updates,
        "transient_call_ids": transient_ids,
    }


def _summary(plan: Dict[str, Any]) -> Dict[str, int]:
    return {
        "affected_sessions": len(plan["affected_sessions"]),
        "assistant_rows_deleted": len(plan["assistant_delete_ids"]),
        "tool_rows_deleted": len(plan["tool_delete_ids"]),
        "mixed_assistant_rows_updated": len(plan["assistant_updates"]),
        "slash_skill_user_rows_cleaned": len(plan["user_updates"]),
        "system_prompts_invalidated": len(plan["stale_prompt_sessions"]),
        "resolved_real_tool_calls": int(plan["resolved_tool_calls"]),
    }


def _backup(profile_home: Path, backup_dir: Path) -> Dict[str, Any]:
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_dir.chmod(0o700)
    source_db = profile_home / "state.db"
    backup_db = backup_dir / "state.db"
    with sqlite3.connect(source_db) as source, sqlite3.connect(backup_db) as target:
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    backup_db.chmod(0o600)
    if integrity != "ok":
        raise RuntimeError(f"backup integrity_check failed: {integrity}")

    config = profile_home / "config.yaml"
    if config.exists():
        shutil.copy2(config, backup_dir / "config.yaml")
        (backup_dir / "config.yaml").chmod(0o600)
    hashes = {
        path.name: _sha256(path)
        for path in backup_dir.iterdir()
        if path.is_file()
    }
    return {"integrity_check": integrity, "sha256": hashes}


def _recount_sessions(conn: sqlite3.Connection, session_ids: Iterable[str]) -> None:
    for session_id in session_ids:
        message_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        tool_rows = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=? AND role='tool'",
            (session_id,),
        ).fetchone()[0]
        tool_calls = 0
        for (raw,) in conn.execute(
            "SELECT tool_calls FROM messages WHERE session_id=? AND tool_calls IS NOT NULL",
            (session_id,),
        ):
            try:
                parsed = json.loads(raw)
                tool_calls += len(parsed) if isinstance(parsed, list) else 1
            except (TypeError, ValueError):
                tool_calls += 1
        conn.execute(
            "UPDATE sessions SET message_count=?, tool_call_count=? WHERE id=?",
            (message_count, tool_rows + tool_calls, session_id),
        )


def apply_plan(conn: sqlite3.Connection, plan: Dict[str, Any]) -> None:
    with conn:
        for row_id, tool_calls in plan["assistant_updates"].items():
            conn.execute(
                "UPDATE messages SET tool_calls=? WHERE id=?",
                (tool_calls, row_id),
            )
        for row_id, (content, api_content) in plan["user_updates"].items():
            conn.execute(
                "UPDATE messages SET content=?, api_content=? WHERE id=?",
                (content, api_content, row_id),
            )
        delete_ids = plan["assistant_delete_ids"] | plan["tool_delete_ids"]
        conn.executemany(
            "DELETE FROM messages WHERE id=?",
            ((row_id,) for row_id in sorted(delete_ids)),
        )
        conn.executemany(
            "UPDATE sessions SET system_prompt=NULL WHERE id=?",
            ((sid,) for sid in sorted(plan["stale_prompt_sessions"])),
        )
        _recount_sessions(conn, plan["affected_sessions"])
        for table in ("messages_fts", "messages_fts_trigram"):
            try:
                conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
            except sqlite3.DatabaseError:
                pass


def migrate_snapshots(profile_home: Path, *, apply: bool) -> int:
    from agent.capability_history import durable_projection

    changed = 0
    for path in (profile_home / "sessions").glob("session_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            messages = payload.get("messages")
            if not isinstance(messages, list):
                continue
            projected = durable_projection(messages)
            for msg in projected:
                if msg.get("role") == "user":
                    msg["content"] = _clean_skill_user_content(msg.get("content"))
                    if "api_content" in msg:
                        msg["api_content"] = _clean_skill_user_content(
                            msg.get("api_content")
                        )
            if projected == messages:
                continue
            changed += 1
            if apply:
                payload["messages"] = projected
                payload["message_count"] = len(projected)
                tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp, path)
        except (OSError, ValueError, TypeError):
            continue
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-home", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()

    profile_home = args.profile_home.resolve()
    db_path = profile_home / "state.db"
    with sqlite3.connect(db_path) as conn:
        plan = analyze(conn)
        summary = _summary(plan)
    summary["json_snapshots_affected"] = migrate_snapshots(
        profile_home, apply=False
    )
    print(json.dumps({"dry_run": summary}, ensure_ascii=False, sort_keys=True))
    if not args.apply:
        return 0
    if args.backup_dir is None:
        parser.error("--backup-dir is required with --apply")

    backup_meta = _backup(profile_home, args.backup_dir)
    with sqlite3.connect(db_path) as conn:
        apply_plan(conn, plan)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"post-migration integrity_check failed: {integrity}")
    migrate_snapshots(profile_home, apply=True)

    manifest = {
        "profile_home": str(profile_home),
        "created_at": time.time(),
        "backup": backup_meta,
        "migration": summary,
        "post_integrity_check": integrity,
    }
    manifest_path = args.backup_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)
    print(json.dumps({"applied": summary, "backup": str(args.backup_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
