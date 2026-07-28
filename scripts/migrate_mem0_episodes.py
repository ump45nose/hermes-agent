#!/usr/bin/env python3
"""Import Episode memories from Mem0 into the canonical local context store.

This command is intentionally separate from the profile/prompt migration.  It
does not rewrite profiles, prompts, MEMORY.md, AGENTS.md, or remote Mem0 data.
The dedicated credential is read from the process environment or the root
``.env`` without printing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MEM0_LIST_URL = "https://api.mem0.ai/v3/memories/"
DEFAULT_ROOT = Path("/home/hermes/.hermes")
PAGE_SIZE = 100
MAX_REMOTE_RECORDS_PER_SUBJECT = 10_000
EPISODE_BODY_FIELDS = (
    "title",
    "goal",
    "context",
    "actions",
    "decisions",
    "outcome",
    "summary",
    "artifacts",
    "open_loops",
    "reusable_lesson",
)


def _install_repo_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    stable_metadata: dict[str, Any] = {}
    for key in (
        "memory_kind",
        "schema_version",
        "source",
        "profile",
        "subject_id",
        "source_session_id",
        "source_hash",
        "outcome",
    ):
        value = metadata.get(key)
        if value is not None:
            # Mem0 currently serializes numeric metadata values as strings.
            # Normalize both local and remote representations so schema_version
            # 2 and "2" do not become a false body conflict.
            stable_metadata[key] = str(value)
    return _sha256(
        _canonical_json(
            {
                "subject_id": record["subject_id"],
                "profile": record["profile"],
                "run_id": record["run_id"],
                "source_hash": record["source_hash"],
                "body": record["body"],
                "metadata": stable_metadata,
            }
        )
    )


def _identity(record: dict[str, Any]) -> str:
    return "\x1f".join(
        (
            record["subject_id"],
            record["run_id"],
            record["source_hash"],
        )
    )


def _safe_identity(record: dict[str, Any]) -> str:
    return " | ".join(
        (
            record["subject_id"],
            record["run_id"],
            record["source_hash"],
        )
    )


def _load_dedicated_key(root: Path, env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    _install_repo_path()
    from agent.secret_scope import load_env_file

    return str(load_env_file(root / ".env").get(env_name) or "").strip()


def _request_page(
    key: str,
    *,
    subject_id: str,
    page: int,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{MEM0_LIST_URL}?page={page}&page_size={PAGE_SIZE}",
        data=json.dumps(
            {
                "filters": {
                    "AND": [
                        {"user_id": f"hermes-episodes:{subject_id}"},
                    ]
                }
            }
        ).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Token {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "hermes-episode-local-migration/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # Do not include the response body: a provider error can reflect request
        # metadata and the migration report must stay credential/body free.
        raise RuntimeError(f"Mem0 list request failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Mem0 list request failed: {type(exc.reason).__name__}") from exc
    value = json.loads(body or "{}")
    if not isinstance(value, dict):
        raise RuntimeError("Mem0 list response was not an object")
    return value


def _remote_snapshot(
    key: str,
    *,
    subject_id: str,
    timeout: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_remote_ids: set[str] = set()
    page = 1
    while len(records) < MAX_REMOTE_RECORDS_PER_SUBJECT:
        response = _request_page(
            key,
            subject_id=subject_id,
            page=page,
            timeout=timeout,
        )
        results = response.get("results") or []
        if not isinstance(results, list):
            raise RuntimeError("Mem0 list response results was not an array")
        new_ids = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            remote_id = str(item.get("id") or _sha256(_canonical_json(item)))
            if remote_id in seen_remote_ids:
                continue
            seen_remote_ids.add(remote_id)
            records.append(item)
            new_ids += 1
        if not results or len(results) < PAGE_SIZE:
            break
        if new_ids == 0:
            raise RuntimeError("Mem0 pagination repeated a full page")
        page += 1
    if len(records) >= MAX_REMOTE_RECORDS_PER_SUBJECT:
        raise RuntimeError(
            f"Mem0 subject exceeded safety cap {MAX_REMOTE_RECORDS_PER_SUBJECT}"
        )
    return records


def _normalize_remote(
    item: dict[str, Any],
    *,
    subject_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None, "metadata_not_object"
    if metadata.get("memory_kind") != "episode":
        return None, None
    source_hash = str(metadata.get("source_hash") or "").strip()
    profile = str(metadata.get("profile") or "").strip()
    source_session_id = str(metadata.get("source_session_id") or "").strip()
    run_id = str(item.get("run_id") or metadata.get("run_id") or "").strip()
    if not run_id and source_session_id and profile:
        # Mem0's list endpoint does not currently echo the request run_id.
        # Reconstruct the canonical id from metadata written by Episode sync
        # before falling back to the remote memory UUID.
        run_id = f"hermes:{subject_id}:{profile}:{source_session_id}"
    if not run_id:
        run_id = str(item.get("id") or "").strip()
    if not source_hash:
        return None, "missing_source_hash"
    if not run_id:
        return None, "missing_run_id"
    if not profile:
        return None, "missing_profile"
    body = item.get("memory")
    if not isinstance(body, str):
        body = _canonical_json(body)
    else:
        try:
            decoded_body = json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded_body = None
        if isinstance(decoded_body, dict):
            # The shadow payload adds episode_schema_version for the remote API;
            # the canonical local row stores only Episode content fields.
            body = json.dumps(
                {
                    key: decoded_body.get(key)
                    for key in EPISODE_BODY_FIELDS
                },
                ensure_ascii=False,
                sort_keys=True,
            )
    record = {
        "remote_id": str(item.get("id") or ""),
        "subject_id": subject_id,
        "profile": profile,
        "run_id": run_id,
        "source_hash": source_hash,
        "body": body,
        "metadata": metadata,
    }
    record["record_hash"] = _record_hash(record)
    return record, None


def _discover_subjects(root: Path) -> list[str]:
    subjects: set[str] = set()
    candidates = [
        root / "episode-sync" / "users",
        *(root / "profiles").glob("*/episode-sync/users"),
    ]
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_dir() and path.name.startswith("user-"):
                subjects.add(path.name)
    return sorted(subjects)


def _backup_database(db_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_dir.chmod(0o700)
    target = backup_dir / "context.db"
    if db_path.is_file():
        source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
    else:
        sqlite3.connect(target).close()
    target.chmod(0o600)
    return target


def _existing_record(
    conn: sqlite3.Connection,
    record: dict[str, Any],
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT profile, body, metadata_json FROM episodes "
        "WHERE subject_id=? AND run_id=? AND source_hash=?",
        (
            record["subject_id"],
            record["run_id"],
            record["source_hash"],
        ),
    ).fetchone()


def _existing_hash(record: dict[str, Any], row: sqlite3.Row) -> str:
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    value = {
        **record,
        "profile": str(row["profile"]),
        "body": str(row["body"]),
        "metadata": metadata,
    }
    return _record_hash(value)


def _digest(records: Iterable[dict[str, Any]]) -> str:
    lines = sorted(
        f"{_identity(record)}\x1f{record['record_hash']}"
        for record in records
    )
    return _sha256("\n".join(lines))


def _samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: (_identity(record), record["record_hash"]))
    if not ordered:
        return []
    indexes = sorted({0, len(ordered) // 2, len(ordered) - 1})
    samples: list[dict[str, Any]] = []
    for index in indexes:
        record = ordered[index]
        samples.append(
            {
                "identity": _safe_identity(record),
                "record_hash": record["record_hash"],
                "body_chars": len(record["body"]),
                "profile": record["profile"],
                "outcome": record["metadata"].get("outcome"),
                "schema_version": record["metadata"].get("schema_version"),
            }
        )
    return samples


def _migrate_subject(
    conn: sqlite3.Connection,
    *,
    subject_id: str,
    remote_items: list[dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for item in remote_items:
        record, reason = _normalize_remote(item, subject_id=subject_id)
        if reason:
            invalid.append(
                {
                    "remote_id": str(item.get("id") or ""),
                    "reason": reason,
                }
            )
        elif record is not None:
            records.append(record)

    inserted = duplicate = 0
    conflicts: list[dict[str, str]] = []
    for record in records:
        existing = _existing_record(conn, record)
        if existing is None:
            conn.execute(
                "INSERT INTO episodes "
                "(subject_id, profile, run_id, source_hash, body, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["subject_id"],
                    record["profile"],
                    record["run_id"],
                    record["source_hash"],
                    record["body"],
                    _canonical_json(record["metadata"]),
                ),
            )
            inserted += 1
            continue
        local_hash = _existing_hash(record, existing)
        if local_hash == record["record_hash"]:
            duplicate += 1
        else:
            conflicts.append(
                {
                    "identity": _safe_identity(record),
                    "remote_record_hash": record["record_hash"],
                    "local_record_hash": local_hash,
                }
            )

    local_rows = conn.execute(
        "SELECT subject_id, profile, run_id, source_hash, body, metadata_json "
        "FROM episodes WHERE subject_id=?",
        (subject_id,),
    ).fetchall()
    local_records: list[dict[str, Any]] = []
    for row in local_rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        local = {
            "subject_id": str(row["subject_id"]),
            "profile": str(row["profile"]),
            "run_id": str(row["run_id"]),
            "source_hash": str(row["source_hash"]),
            "body": str(row["body"]),
            "metadata": metadata,
        }
        local["record_hash"] = _record_hash(local)
        local_records.append(local)

    remote_by_identity = {_identity(record): record for record in records}
    local_by_identity = {_identity(record): record for record in local_records}
    missing = sorted(set(remote_by_identity) - set(local_by_identity))
    extra = sorted(set(local_by_identity) - set(remote_by_identity))
    matched_remote = [
        record
        for identity, record in remote_by_identity.items()
        if identity in local_by_identity
        and record["record_hash"] == local_by_identity[identity]["record_hash"]
    ]
    return {
        "subject_id": subject_id,
        "remote_total": len(remote_items),
        "remote_eligible": len(records),
        "inserted": inserted,
        "duplicate": duplicate,
        "invalid": invalid,
        "conflicts": conflicts,
        "local_total": len(local_records),
        "matched": len(matched_remote),
        "missing": missing,
        "extra": extra,
        "remote_digest": _digest(records),
        "local_digest": _digest(local_records),
        "local_matched_digest": _digest(matched_remote),
        "samples": _samples(records),
    }


def migrate(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    root = args.root.resolve()
    db_path = root / "user-context" / "context.db"
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    backup_dir = root / "backups" / "episode-mem0-migration" / stamp
    key = _load_dedicated_key(root, args.key_env)
    if not key:
        raise RuntimeError(f"{args.key_env} is not configured")
    subjects = sorted(set(args.subject or _discover_subjects(root)))
    if not subjects:
        raise RuntimeError("No Episode subjects were discovered")

    snapshots: dict[str, list[dict[str, Any]]] = {}
    for subject_id in subjects:
        snapshots[subject_id] = _remote_snapshot(
            key,
            subject_id=subject_id,
            timeout=args.timeout,
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.chmod(0o700)
    _backup_database(db_path, backup_dir)
    _install_repo_path()
    from agent.local_context import LocalContextStore

    store = LocalContextStore(db_path)
    with store.connect() as conn:
        conn.row_factory = sqlite3.Row
        subject_reports = [
            _migrate_subject(
                conn,
                subject_id=subject_id,
                remote_items=snapshots[subject_id],
            )
            for subject_id in subjects
        ]
        complete = all(
            not report["invalid"]
            and not report["conflicts"]
            and not report["missing"]
            and not report["extra"]
            and report["remote_eligible"] == report["matched"]
            and report["local_total"] == report["matched"]
            and report["remote_digest"] == report["local_digest"]
            for report in subject_reports
        )
        summary = {
            "migration_started_at": started_at.isoformat(),
            "migration_finished_at": datetime.now(timezone.utc).isoformat(),
            "subjects": subject_reports,
            "totals": {
                key: sum(int(report[key]) for report in subject_reports)
                for key in (
                    "remote_total",
                    "remote_eligible",
                    "inserted",
                    "duplicate",
                    "local_total",
                    "matched",
                )
            },
            "global_remote_digest": _sha256(
                "\n".join(
                    f"{report['subject_id']}:{report['remote_digest']}"
                    for report in subject_reports
                )
            ),
            "complete": complete,
            "remote_deleted": False,
            "backup": str(backup_dir / "context.db"),
        }
        conn.execute(
            "INSERT INTO sync_state(key, value_json) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
            "updated_at=CURRENT_TIMESTAMP",
            ("mem0_episode_migration", _canonical_json(summary)),
        )
        conn.execute(
            "INSERT INTO audit_events(event, detail_json) VALUES (?, ?)",
            ("mem0_episode_migration", _canonical_json(summary)),
        )

    report_path = backup_dir / "migration-report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.chmod(0o600)
    return summary, complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subject", action="append")
    parser.add_argument("--key-env", default="MEM0_EPISODES_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main() -> int:
    try:
        report, complete = migrate(build_parser().parse_args())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
