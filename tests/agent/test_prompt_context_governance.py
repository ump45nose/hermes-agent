import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from agent.local_context import LocalContextStore
from agent.episode_policy import eligible_session, episode_input_messages
from agent.request_snapshot import capture_request_snapshot
from agent.runtime_role import resolve_runtime_role
from agent.tool_context_editor import edit_tool_context
from hermes_cli.prompt_compiler import (
    compile_profile_prompt,
    load_compiled_prompt,
    verify_compiled_prompt,
    write_compiled_prompt,
)
from hermes_constants import get_canonical_hermes_root, get_profile_home
from hermes_state import SessionDB


def test_canonical_root_does_not_repeat_named_profile():
    home = Path("/srv/hermes/profiles/lingjun")
    assert get_canonical_hermes_root(home) == Path("/srv/hermes")
    assert get_profile_home("lingjun", root=get_canonical_hermes_root(home)) == home


def test_compiled_prompt_is_fixed_and_verifiable(tmp_path):
    lock = write_compiled_prompt(
        tmp_path,
        preset="research",
        model_family="openai",
    )
    loaded = load_compiled_prompt(tmp_path)
    assert loaded is not None
    text, saved_lock = loaded
    assert saved_lock["compiled_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert lock["model_adapter"]["family"] == "openai"
    assert verify_compiled_prompt(tmp_path)["ok"]
    assert "Git" not in text
    assert "依赖安装" not in text


def _worker_db(path: Path, *, claim: str = "claim", expires: float | None = None):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE tasks (id TEXT, status TEXT, current_run_id INTEGER, "
            "claim_lock TEXT, claim_expires REAL)"
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
            ("task-1", "running", 7, claim, expires or time.time() + 300),
        )


def test_worker_role_requires_valid_lease(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    _worker_db(db)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "claim")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    role = resolve_runtime_role("kanban_worker")
    assert role.role == "kanban_worker"
    assert role.verified

    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "wrong")
    failed = resolve_runtime_role("kanban_worker")
    assert failed.role == "interactive"
    assert not failed.verified


def _assistant(call_id: str, name: str = "web_search"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool(call_id: str, content: str, *, status: str = "success"):
    return {
        "role": "tool",
        "name": "web_search",
        "tool_name": "web_search",
        "tool_call_id": call_id,
        "content": content,
        "_tool_receipt": {
            "tool_name": "web_search",
            "result_status": status,
            "effect": "none",
            "artifact_ref": None,
            "consumed_turn": 1,
            "steer_present": False,
            "supersedes": None,
        },
    }


def test_tool_editor_preserves_current_and_removes_consumed_duplicate_pair():
    messages = [
        {"role": "user", "content": "search"},
        _assistant("old"),
        _tool("old", "same"),
        _assistant("new"),
        _tool("new", "same"),
    ]
    edited, report = edit_tool_context(messages)
    assert all(msg.get("tool_call_id") != "old" for msg in edited)
    assert any(msg.get("tool_call_id") == "new" for msg in edited)
    assert report[0]["action"] == "remove_pair"


def test_tool_editor_keeps_steer_and_unknown_effect():
    message = _tool("old", "/steer stop")
    message["_tool_receipt"]["steer_present"] = True
    message["_tool_receipt"]["effect"] = "unknown"
    edited, report = edit_tool_context([_assistant("old"), message, {"role": "assistant", "content": "ok"}])
    assert edited == [_assistant("old"), message, {"role": "assistant", "content": "ok"}]
    assert report == []


def test_tool_editor_active_replaces_consumed_side_effect_with_receipt():
    message = _tool("old", "created external object")
    message["_tool_receipt"]["effect"] = "unknown"
    edited, report = edit_tool_context(
        [_assistant("old", "send_message"), message, {"role": "assistant", "content": "done"}],
        phase="active",
    )
    assert "tool action receipt" in edited[1]["content"]
    assert report[0]["action"] == "action_receipt"


def test_local_scenario_keyword_scope_cap_and_session_dedupe(tmp_path):
    store = LocalContextStore(tmp_path / "context.db")
    store.upsert_card(
        card_id="card-1",
        subject_id="subject",
        profile="lingjun",
        title="RSS",
        body="Use the approved RSS workflow.",
        keywords=["RSS 监控", "feed"],
    )
    hits = store.search(
        "检查 RSS 监控",
        subject_id="subject",
        profile="lingjun",
        session_id="session",
    )
    assert [hit.card_id for hit in hits] == ["card-1"]
    assert store.search(
        "RSS 监控",
        subject_id="subject",
        profile="lingjun",
        session_id="session",
    ) == []
    assert store.search(
        "RSS 监控",
        subject_id="other",
        profile="lingjun",
        session_id="other-session",
    ) == []


def test_request_snapshot_once_writes_body_without_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "observability:\n  request_snapshot: once\n", encoding="utf-8"
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        base_url="https://example.test/v1",
        tools=[{"type": "function"}],
        _prompt_source_manifest=[{"kind": "compiled_prompt"}],
    )
    path = capture_request_snapshot(
        agent,
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "extra_headers": {"Authorization": "Bearer secret"},
        },
        request_id="req-1",
    )
    assert path is not None
    payload = json.loads(path.read_text())
    assert "extra_headers" not in payload["body"]
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["observability"]["request_snapshot"] == "off"


def test_tool_receipt_round_trips_session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    session_id = "session-1"
    db.create_session(session_id, source="cli")
    receipt = {
        "tool_name": "web_search",
        "result_status": "success",
        "effect": "none",
        "artifact_ref": "/artifact",
        "consumed_turn": 1,
        "steer_present": False,
        "supersedes": None,
    }
    db.append_message(
        session_id,
        "tool",
        "result",
        tool_name="web_search",
        tool_call_id="call-1",
        effect_disposition="none",
        tool_receipt=receipt,
    )
    messages = db.get_messages_as_conversation(session_id)
    assert messages[0]["_tool_receipt"] == receipt


def test_episode_policy_excludes_tests_and_raw_tool_results():
    assert not eligible_session(
        source="cron",
        session_id="session",
        ended_at=1,
        message_count=20,
    )
    assert not eligible_session(
        source="telegram",
        session_id="pytest-fixture-1",
        ended_at=1,
        message_count=20,
    )
    rows = [
        {"role": "user", "content": "do it"},
        {"role": "tool", "content": "raw secret output", "tool_receipt": None},
        {
            "role": "tool",
            "content": "large raw output",
            "tool_receipt": json.dumps(
                {
                    "tool_name": "web_search",
                    "result_status": "success",
                    "effect": "none",
                    "artifact_ref": "/artifact",
                }
            ),
        },
    ]
    rendered = episode_input_messages(rows)
    assert "raw secret output" not in json.dumps(rendered)
    assert "large raw output" not in json.dumps(rendered)
    assert "/artifact" in json.dumps(rendered)
