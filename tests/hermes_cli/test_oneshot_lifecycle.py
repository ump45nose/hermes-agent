"""Lifecycle coverage for ``hermes -z`` agent ownership."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from hermes_cli import oneshot


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _stub_runtime(monkeypatch, agent_cls, session_db) -> None:
    monkeypatch.setattr(
        oneshot,
        "_create_session_db_for_oneshot",
        lambda: session_db,
    )
    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        _module("run_agent", AIAgent=agent_cls),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module(
            "hermes_cli.config",
            load_config=lambda: {"model": {"default": "test-model"}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module(
            "hermes_cli.models",
            detect_provider_for_model=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "test-key",
                "base_url": "https://example.invalid",
                "provider": "test-provider",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: set(),
        ),
    )


def test_run_agent_closes_agent_after_success(monkeypatch):
    lifecycle: list[str] = []

    class FakeAgent:
        def __init__(self, **_kwargs):
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt):
            lifecycle.append(f"run:{prompt}")
            return {
                "final_response": "kept final response",
                "total_tokens": 17,
            }

        def close(self):
            lifecycle.append("close")

    _stub_runtime(monkeypatch, FakeAgent, object())

    response, result = oneshot._run_agent("hello")

    assert lifecycle == ["run:hello", "close"]
    assert response == "kept final response"
    assert result["total_tokens"] == 17


def test_run_agent_closes_agent_when_conversation_raises(monkeypatch):
    lifecycle: list[str] = []

    class FakeAgent:
        def __init__(self, **_kwargs):
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, _prompt):
            lifecycle.append("run")
            raise RuntimeError("provider failed")

        def close(self):
            lifecycle.append("close")

    _stub_runtime(monkeypatch, FakeAgent, object())

    with pytest.raises(RuntimeError, match="provider failed"):
        oneshot._run_agent("hello")

    assert lifecycle == ["run", "close"]


def test_run_agent_close_persists_session_end(monkeypatch, tmp_path):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    session_db = SessionDB(tmp_path / "state.db")
    session_id = "oneshot-lifecycle-test"

    class SessionOwningAgent:
        def __init__(self, *, session_db, **_kwargs):
            self._session_db = session_db
            self.session_id = session_id
            self._end_session_on_close = True
            self._active_children_lock = threading.RLock()
            self._active_children = []
            self._session_messages = []
            self.client = None
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, _prompt):
            self._session_db.create_session(self.session_id, "cli")
            return {"final_response": "done"}

        def close(self):
            AIAgent.close(self)

    _stub_runtime(monkeypatch, SessionOwningAgent, session_db)

    try:
        response, _result = oneshot._run_agent("persist this session")
        row = session_db.get_session(session_id)
    finally:
        session_db.close()

    assert response == "done"
    assert row is not None
    assert row["ended_at"] is not None
    assert row["end_reason"] == "agent_close"
