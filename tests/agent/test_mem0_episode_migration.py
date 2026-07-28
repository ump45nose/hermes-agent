import copy
import json
import sqlite3

from agent.local_context import LocalContextStore
from scripts.migrate_mem0_episodes import (
    _migrate_subject,
    _normalize_remote,
    _record_hash,
)


def _remote_item(
    *,
    body="episode body",
    synced_at="2026-07-28T00:00:00Z",
    include_run_id=True,
):
    item = {
        "id": "remote-1",
        "memory": body,
        "metadata": {
            "memory_kind": "episode",
            "schema_version": 2,
            "source": "hermes_episode_daily_sync",
            "profile": "lingjun",
            "subject_id": "subject",
            "source_session_id": "session-1",
            "source_hash": "source-hash",
            "outcome": "success",
            "synced_at": synced_at,
        },
    }
    if include_run_id:
        item["run_id"] = "hermes:subject:lingjun:session-1"
    return item


def test_record_hash_ignores_volatile_sync_timestamp():
    first, error = _normalize_remote(_remote_item(), subject_id="subject")
    assert error is None
    second_item = _remote_item(synced_at="2026-07-29T00:00:00Z")
    second, error = _normalize_remote(second_item, subject_id="subject")
    assert error is None
    assert _record_hash(first) == _record_hash(second)


def test_normalize_remote_rejects_episode_without_source_hash():
    item = _remote_item()
    del item["metadata"]["source_hash"]
    record, error = _normalize_remote(item, subject_id="subject")
    assert record is None
    assert error == "missing_source_hash"


def test_normalize_remote_reconstructs_run_id_when_mem0_omits_it():
    record, error = _normalize_remote(
        _remote_item(include_run_id=False),
        subject_id="subject",
    )
    assert error is None
    assert record["run_id"] == "hermes:subject:lingjun:session-1"


def test_normalize_remote_maps_shadow_payload_to_canonical_local_body():
    exported = {
        "episode_schema_version": 2,
        "title": "Title",
        "goal": "Goal",
        "context": [],
        "actions": ["searched"],
        "decisions": [],
        "outcome": "success",
        "summary": "Done",
        "artifacts": [],
        "open_loops": [],
        "reusable_lesson": "",
    }
    record, error = _normalize_remote(
        _remote_item(body=json.dumps(exported), include_run_id=False),
        subject_id="subject",
    )
    assert error is None
    expected = dict(exported)
    expected.pop("episode_schema_version")
    assert json.loads(record["body"]) == expected


def test_record_hash_normalizes_numeric_metadata_strings():
    first, error = _normalize_remote(_remote_item(), subject_id="subject")
    assert error is None
    second = copy.deepcopy(first)
    first["metadata"]["schema_version"] = 2
    second["metadata"]["schema_version"] = "2"
    assert _record_hash(first) == _record_hash(second)


def test_migrate_subject_is_idempotent_and_reports_body_conflict(tmp_path):
    store = LocalContextStore(tmp_path / "context.db")
    with store.connect() as conn:
        conn.row_factory = sqlite3.Row
        first = _migrate_subject(
            conn,
            subject_id="subject",
            remote_items=[_remote_item()],
        )
        assert first["inserted"] == 1
        assert first["missing"] == []
        second = _migrate_subject(
            conn,
            subject_id="subject",
            remote_items=[_remote_item()],
        )
        assert second["duplicate"] == 1
        changed = copy.deepcopy(_remote_item(body="different body"))
        conflict = _migrate_subject(
            conn,
            subject_id="subject",
            remote_items=[changed],
        )
        assert len(conflict["conflicts"]) == 1
