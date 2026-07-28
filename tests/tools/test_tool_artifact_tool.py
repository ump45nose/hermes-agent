import json
from pathlib import Path

from tools import tool_artifact_tool


def _artifact(tmp_path: Path, session_id: str, text: str) -> Path:
    path = (
        tmp_path
        / "artifacts"
        / "tool-results"
        / session_id
        / "call-1.txt"
    )
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
        )
    )
    assert result["ok"] is False
    assert result["error_type"] == "unsafe_permissions"
