import json
from pathlib import Path

from hermes_constants import contained_session_path
from tools import tool_artifact_tool


def _artifact(tmp_path: Path, session_id: str, text: str) -> Path:
    path = contained_session_path(
        tmp_path / "artifacts" / "tool-results",
        session_id,
    )
    path = path / "call-1.txt"
    path.parent.mkdir(parents=True)
    path.parent.chmod(0o700)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_read_tool_artifact_returns_bounded_page(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_artifact_tool, "get_hermes_home", lambda: tmp_path)
    path = _artifact(tmp_path, "session-a", "0123456789")
    result = json.loads(
        tool_artifact_tool.read_tool_artifact(
            {"artifact_ref": str(path), "offset": 3, "limit": 4},
            session_id="session-a",
            runtime_role="research_leaf",
        )
    )
    assert result["ok"] is True
    assert result["content"] == "3456"
    assert result["next_offset"] == 7
    assert result["total_chars"] == 10


def test_read_tool_artifact_rejects_cross_session_path(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_artifact_tool, "get_hermes_home", lambda: tmp_path)
    path = _artifact(tmp_path, "session-b", "private")
    result = json.loads(
        tool_artifact_tool.read_tool_artifact(
            {"artifact_ref": str(path)},
            session_id="session-a",
            runtime_role="research_leaf",
        )
    )
    assert result["ok"] is False
    assert result["error_type"] == "scope_violation"


def test_read_tool_artifact_rejects_non_owner_only_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_artifact_tool, "get_hermes_home", lambda: tmp_path)
    path = _artifact(tmp_path, "session-a", "private")
    path.chmod(0o640)
    result = json.loads(
        tool_artifact_tool.read_tool_artifact(
            {"artifact_ref": str(path)},
            session_id="session-a",
            runtime_role="research_leaf",
        )
    )
    assert result["ok"] is False
    assert result["error_type"] == "unsafe_permissions"


def test_read_tool_artifact_rejects_non_research_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_artifact_tool, "get_hermes_home", lambda: tmp_path)
    path = _artifact(tmp_path, "session-a", "private")
    result = json.loads(
        tool_artifact_tool.read_tool_artifact(
            {"artifact_ref": str(path)},
            session_id="session-a",
            runtime_role="interactive",
        )
    )
    assert result["ok"] is False
    assert result["error_type"] == "runtime_scope_violation"


def test_absolute_session_id_cannot_redefine_artifact_root(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_artifact_tool, "get_hermes_home", lambda: tmp_path)
    outside = tmp_path / "outside" / "private.txt"
    outside.parent.mkdir()
    outside.write_text("private", encoding="utf-8")
    outside.chmod(0o600)

    result = json.loads(
        tool_artifact_tool.read_tool_artifact(
            {"artifact_ref": str(outside)},
            session_id=str(outside.parent),
            runtime_role="research_leaf",
        )
    )
    assert result["ok"] is False
    assert result["error_type"] == "scope_violation"


def test_traversal_session_id_is_normalized_under_artifact_root(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tool_artifact_tool, "get_hermes_home", lambda: tmp_path)
    raw_session_id = "../../session"
    path = _artifact(tmp_path, raw_session_id, "bounded")
    canonical_root = (tmp_path / "artifacts" / "tool-results").resolve()
    assert path.parent.parent == canonical_root
    assert ".." not in path.parent.name

    result = json.loads(
        tool_artifact_tool.read_tool_artifact(
            {"artifact_ref": str(path)},
            session_id=raw_session_id,
            runtime_role="research_leaf",
        )
    )
    assert result["ok"] is True
    assert result["content"] == "bounded"


def test_read_tool_artifact_rejects_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_artifact_tool, "get_hermes_home", lambda: tmp_path)
    session_root = contained_session_path(
        tmp_path / "artifacts" / "tool-results",
        "session-a",
    )
    session_root.mkdir(parents=True)
    session_root.chmod(0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    outside.chmod(0o600)
    link = session_root / "call-link.txt"
    link.symlink_to(outside)

    result = json.loads(
        tool_artifact_tool.read_tool_artifact(
            {"artifact_ref": str(link)},
            session_id="session-a",
            runtime_role="research_leaf",
        )
    )
    assert result["ok"] is False
    assert result["error_type"] == "scope_violation"
