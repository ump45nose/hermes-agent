import pytest

from hermes_cli.tools_config import _get_platform_tool_exposure, _get_platform_tools


def _config(platform_value):
    return {
        "tools": {
            "disclosure": {"mode": "progressive", "schema_scope": "turn"},
        },
        "platform_toolsets": {"telegram": platform_value},
        "mcp_servers": {"notion": {"url": "https://example.invalid/mcp"}},
    }


def test_structured_platform_union_is_reachable():
    cfg = _config({
        "direct": [],
        "deferred": ["file", "notion"],
    })
    exposure = _get_platform_tool_exposure(cfg, "telegram")
    assert exposure.progressive
    assert exposure.direct == frozenset()
    assert exposure.deferred == frozenset({"file", "notion"})
    assert _get_platform_tools(cfg, "telegram") == {"file", "notion"}


def test_structured_overlap_fails_closed():
    cfg = _config({
        "direct": ["file"],
        "deferred": ["file"],
    })
    with pytest.raises(ValueError, match="overlap"):
        _get_platform_tools(cfg, "telegram")


def test_structured_unknown_capability_fails_closed():
    cfg = _config({
        "direct": [],
        "deferred": ["definitely-not-a-toolset"],
    })
    with pytest.raises(ValueError, match="unknown capabilities"):
        _get_platform_tools(cfg, "telegram")


def test_unlisted_v2_platform_has_no_functional_capabilities():
    cfg = _config({"direct": [], "deferred": ["file"]})
    exposure = _get_platform_tool_exposure(cfg, "unlisted")
    assert exposure.progressive
    assert exposure.reachable == frozenset()
    assert _get_platform_tools(cfg, "unlisted") == set()


def test_worker_only_kanban_requires_verified_lease(monkeypatch, tmp_path):
    import sqlite3
    import time

    db = tmp_path / "kanban.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE tasks (id TEXT, status TEXT, current_run_id INTEGER, "
            "claim_lock TEXT, claim_expires REAL)"
        )
        conn.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
            ("task-1", "running", 9, "claim", time.time() + 300),
        )
    cfg = _config({"direct": [], "deferred": ["file"]})
    cfg["tools"]["kanban"] = {"worker_only": True}
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "9")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "claim")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    assert "kanban" in _get_platform_tools(cfg, "telegram")
