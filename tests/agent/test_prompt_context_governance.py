import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent.local_context import LocalContextStore
from agent.episode_policy import eligible_session, episode_input_messages
from agent.request_snapshot import capture_request_snapshot
from agent.runtime_role import (
    RESEARCH_LEAF_PROMPT,
    resolve_runtime_role,
    runtime_capability_overlay,
    scope_runtime_toolsets,
)
from agent.tool_context_editor import (
    edit_tool_context,
    mark_tool_results_consumed,
    receipt_from_message,
    strip_internal_tool_metadata,
)
from agent.tool_result_classification import tool_may_have_side_effect
from agent.prompt_builder import format_steer_marker
from hermes_cli.prompt_compiler import (
    compile_profile_prompt,
    extra_modules_from_lock,
    load_compiled_prompt,
    render_prompt_diff,
    validate_runtime_prompt_protocols,
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
    assert "git commit" not in text.lower()
    assert "编辑代码" not in text
    assert "依赖安装" not in text


def test_controller_lock_v2_provisions_protocol_capabilities(tmp_path):
    lock = write_compiled_prompt(
        tmp_path,
        preset="lingjun",
        model_family="openai",
    )
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())

    assert lock["schema_version"] == 2
    assert lock["protocols"] == ["kanban-controller@1"]
    assert lock["runtime_overlays"] == ["platform", "cron"]
    for surface in ("cli", "telegram", "weixin", "api_server"):
        assert "kanban" in config["platform_toolsets"][surface]["deferred"]
    assert verify_compiled_prompt(tmp_path)["ok"]

    runtime = validate_runtime_prompt_protocols(
        lock,
        platform="telegram",
        runtime_role="interactive",
        reachable_toolsets={"kanban"},
    )
    assert runtime["ok"]
    assert runtime["protocols"] == ["kanban-controller@1"]

    missing = validate_runtime_prompt_protocols(
        lock,
        platform="telegram",
        runtime_role="interactive",
        reachable_toolsets=set(),
    )
    assert not missing["ok"]
    assert missing["missing"] == ["kanban.create", "kanban.roster"]


def test_controller_module_is_generic_control_protocol_not_incident_patch():
    text, lock = compile_profile_prompt("lingjun")

    assert "直接处理、明确派单或进入 Triage" in text
    assert "Harness 负责能力校验" in text
    assert "多来源调查：交给研究能力" in text
    for incident_term in ("GitHub MCP", "市场/JD", "样本描述成频率"):
        assert incident_term not in text
    controller = next(
        module for module in lock["modules"] if module["id"] == "controller"
    )
    assert controller["version"] == 4
    assert controller["protocols"] == ["kanban-controller@1"]


def test_research_lock_requires_delegation_and_rejects_unlisted_overlay(tmp_path):
    lock = write_compiled_prompt(tmp_path, preset="research")
    assert lock["protocols"] == ["research-parent@1"]
    assert "github.read" in lock["required_capabilities"]
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    github = config["mcp_servers"]["github"]
    include = set(github["tools"]["include"])
    declared_read_only = set(github["tool_effects"]["read_only"])
    assert github["enabled"] is True
    assert "get_file_contents" in include
    assert "search_repositories" in include
    assert "create_or_update_file" not in include
    assert include == declared_read_only
    assert "github" in config["platform_toolsets"]["subagent"]["deferred"]
    assert "github" in config["delegation"]["research_leaf_toolsets"]
    runtime = validate_runtime_prompt_protocols(
        lock,
        platform="cron",
        runtime_role="cron",
        reachable_toolsets={"delegation", "github"},
    )
    assert runtime["ok"]

    forbidden = validate_runtime_prompt_protocols(
        lock,
        platform="telegram",
        runtime_role="subagent",
        reachable_toolsets={"delegation", "github"},
    )
    # A generic subagent overlay is not declared by this fixed Profile.
    assert not forbidden["ok"]
    assert forbidden["missing"] == ["runtime_overlay:subagent"]


def test_prompt_upgrade_preserves_locked_extra_modules_and_shows_config_diff(
    tmp_path,
):
    write_compiled_prompt(
        tmp_path,
        preset="lingjun",
        extra_modules=("citation-rigor",),
    )
    loaded = load_compiled_prompt(tmp_path)
    assert loaded is not None
    assert extra_modules_from_lock(loaded[1]) == ("citation-rigor",)

    # Simulate an old config which has lost the protocol dependency.
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    diff = render_prompt_diff(tmp_path)
    assert "config.yaml.new" in diff
    assert "kanban" in diff


def test_fixed_prompt_runtime_shell_carries_capability_version(monkeypatch):
    import agent.system_prompt as system_prompt

    monkeypatch.setattr(
        system_prompt,
        "_ra",
        lambda: SimpleNamespace(load_soul_md=lambda _limit: "SOUL"),
    )
    agent = SimpleNamespace(
        runtime_role="interactive",
        _progressive_disclosure=True,
        _compiled_prompt="compiled duties",
        _prompt_lock={"compiled_sha256": "locked"},
        platform="telegram",
        _memory_store=None,
        _memory_manager=None,
        _scenario_context="",
        model="model-a",
        provider="provider-a",
    )
    parts = system_prompt._build_fixed_profile_prompt_parts(agent, None)
    assert parts["stable"].startswith("Capability-Prompt-Version: 2\n\n")
    assert "compiled duties" in parts["stable"]
    assert agent._prompt_source_manifest[0] == {
        "kind": "capability_protocol",
        "version": 2,
    }


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


def test_research_leaf_runtime_overlay_only_adds_scoped_artifact_reader():
    direct, deferred = runtime_capability_overlay(
        "research_leaf",
        direct=set(),
        deferred={"web", "smart-search", "tool_artifact"},
    )
    assert direct == frozenset({"tool_artifact"})
    assert deferred == frozenset({"web", "smart-search"})

    interactive_direct, interactive_deferred = runtime_capability_overlay(
        "interactive",
        direct={"tool_artifact"},
        deferred={"web", "tool_artifact"},
    )
    assert interactive_direct == frozenset()
    assert interactive_deferred == frozenset({"web"})


def test_non_research_toolset_scope_removes_artifact_from_default_all_and_explicit():
    enabled, disabled = scope_runtime_toolsets(
        "interactive",
        enabled=None,
        disabled=[],
    )
    assert enabled is None
    assert disabled == ["tool_artifact"]

    enabled, disabled = scope_runtime_toolsets(
        "kanban_worker",
        enabled=["web", "tool_artifact"],
        disabled=["terminal"],
    )
    assert enabled == ["web"]
    assert disabled == ["terminal", "tool_artifact"]


def test_research_toolset_scope_retains_artifact_reader():
    enabled, disabled = scope_runtime_toolsets(
        "research_leaf",
        enabled=["web", "tool_artifact"],
        disabled=["tool_artifact", "terminal"],
    )
    assert enabled == ["web", "tool_artifact"]
    assert disabled == ["terminal"]


def test_research_leaf_spills_large_smart_search_before_first_injection():
    from agent.tool_executor import _budget_for_agent

    leaf = SimpleNamespace(
        runtime_role="research_leaf",
        context_compressor=SimpleNamespace(context_length=200_000),
    )
    budget = _budget_for_agent(leaf)
    assert budget.resolve_threshold("mcp__smart_search__smart_search") == 16_000
    assert budget.resolve_threshold("mcp__smart_search__smart_fetch") == 16_000
    assert budget.resolve_threshold("mcp__smart_search__smart_doctor") == 8_000
    assert budget.preview_size == 4_000

    interactive = SimpleNamespace(
        runtime_role="interactive",
        context_compressor=SimpleNamespace(context_length=200_000),
    )
    assert (
        _budget_for_agent(interactive).resolve_threshold(
            "mcp__smart_search__smart_search"
        )
        == 100_000
    )

    readonly_parent = SimpleNamespace(
        runtime_role="interactive",
        _tool_context_editor_mode="readonly",
        context_compressor=SimpleNamespace(context_length=200_000),
    )
    assert (
        _budget_for_agent(readonly_parent).resolve_threshold(
            "mcp__smart_search__smart_fetch"
        )
        == 16_000
    )

    failures_parent = SimpleNamespace(
        runtime_role="interactive",
        _tool_context_editor_mode="failures",
        context_compressor=SimpleNamespace(context_length=200_000),
    )
    assert (
        _budget_for_agent(failures_parent).resolve_threshold(
            "mcp__smart_search__smart_fetch"
        )
        == 16_000
    )
    assert (
        _budget_for_agent(failures_parent).resolve_threshold("session_search")
        == 16_000
    )

    active_parent = SimpleNamespace(
        runtime_role="interactive",
        _tool_context_editor_mode="active",
        context_compressor=SimpleNamespace(context_length=200_000),
    )
    assert (
        _budget_for_agent(active_parent).resolve_threshold(
            "mcp__smart_search__smart_fetch"
        )
        == 16_000
    )

    report_only_interactive = SimpleNamespace(
        runtime_role="interactive",
        _tool_context_editor_mode="report_only",
        context_compressor=SimpleNamespace(context_length=200_000),
    )
    assert (
        _budget_for_agent(report_only_interactive).resolve_threshold(
            "mcp__smart_search__smart_fetch"
        )
        == 100_000
    )


def _assistant(
    call_id: str,
    name: str = "web_search",
    arguments: dict | None = None,
):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments or {}),
                },
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
    assert any(item["action"] == "remove_pair" for item in report)


def test_tool_receipt_is_backward_and_forward_compatible():
    old_receipt = receipt_from_message(
        {
            "role": "tool",
            "content": "legacy",
            "_tool_receipt": {"tool_name": "web_search"},
        }
    )
    assert old_receipt.tool_name == "web_search"
    assert old_receipt.request_ledger is None
    assert old_receipt.result_status == "unknown"

    future_receipt = receipt_from_message(
        {
            "role": "tool",
            "content": "future",
            "_tool_receipt": {
                "tool_name": "web_search",
                "request_ledger": {"query": "kept"},
                "future_field": {"ignored": True},
            },
        }
    )
    assert future_receipt.request_ledger == {"query": "kept"}
    assert "future_field" not in future_receipt.to_dict()


def test_tool_editor_receiptizes_unique_consumed_read_result():
    messages = [
        {"role": "user", "content": "search"},
        _assistant("old"),
        _tool("old", "unique useful evidence"),
        {"role": "assistant", "content": "I used that evidence."},
    ]
    edited, report = edit_tool_context(messages, phase="readonly")
    assert len(edited) == len(messages)
    assert "body cleared" in edited[2]["content"]
    assert "sha256=" in edited[2]["content"]
    assert report == [
        {
            "tool_call_id": "old",
            "tool_name": "web_search",
            "action": "read_receipt",
            "status": "success",
            "artifact_ref": None,
        }
    ]


def test_smart_search_receipt_keeps_source_index_not_result_body():
    secret = "sk-proj-" + "a" * 30
    nested = {
        "ok": True,
        "query": "RSS readers",
        "content": "raw evidence body that must be cleared",
        "sources_count": 2,
        "providers_used": ["example"],
        "sources": [
            {
                "title": f"Official documentation {secret}",
                "url": "https://example.test/docs",
                "provider": "example",
            },
            {
                "title": "Release notes",
                "url": "https://example.test/releases",
                "provider": "example",
            },
        ],
    }
    content = (
        '<untrusted_tool_result source="mcp__smart_search__smart_search">\n'
        + json.dumps({"result": json.dumps(nested)})
        + "\n</untrusted_tool_result>"
    )
    message = _tool("rss", content)
    message["name"] = message["tool_name"] = "mcp__smart_search__smart_search"
    message["_tool_receipt"].update(
        {
            "tool_name": "mcp__smart_search__smart_search",
            "artifact_ref": "/owner/session/rss.txt",
        }
    )
    edited, report = edit_tool_context(
        [
            _assistant("rss", "mcp__smart_search__smart_search"),
            message,
            {"role": "assistant", "content": "Evidence recorded."},
        ],
        phase="readonly",
    )
    receipt = edited[1]["content"]
    assert "raw evidence body that must be cleared" not in receipt
    assert "RSS readers" in receipt
    assert "https://example.test/docs" in receipt
    assert '"sources_count":2' in receipt
    assert secret not in receipt
    assert "/owner/session/rss.txt" in receipt
    assert len(receipt) < 3_500
    assert report[0]["action"] == "read_receipt"


def test_smart_search_receipt_keeps_owner_request_ledger_after_pre_spill():
    content = (
        "Preview only; original body is not present.\n"
        "Full output saved to: /owner/session/fetch.txt"
    )
    message = _tool("fetch", content)
    message["name"] = message["tool_name"] = "mcp__smart_search__smart_fetch"
    message["_tool_receipt"].update(
        {
            "tool_name": "mcp__smart_search__smart_fetch",
            "artifact_ref": "/owner/session/fetch.txt",
        }
    )
    edited, report = edit_tool_context(
        [
            _assistant(
                "fetch",
                "mcp__smart_search__smart_fetch",
                {
                    "url": "https://example.test/article",
                    "query": "rss reader comparison",
                    "question": "Which claims are verified?",
                    "raw_body": "must-not-enter-ledger",
                },
            ),
            message,
            {"role": "assistant", "content": "Evidence consumed."},
        ],
        phase="readonly",
    )
    receipt = edited[1]["content"]
    assert "REQUEST LEDGER" in receipt
    assert "https://example.test/article" in receipt
    assert "rss reader comparison" in receipt
    assert "Which claims are verified?" in receipt
    assert "must-not-enter-ledger" not in receipt
    assert "Full output saved to" not in receipt
    assert report[0]["action"] == "read_receipt"


def test_tool_editor_report_only_reports_read_receipt_without_mutation():
    messages = [
        _assistant("old"),
        _tool("old", "unique useful evidence"),
        {"role": "assistant", "content": "done"},
    ]
    edited, report = edit_tool_context(
        messages,
        report_only=True,
        phase="readonly",
    )
    assert edited is messages
    assert edited[1]["content"] == "unique useful evidence"
    assert report[0]["action"] == "read_receipt"


def test_tool_editor_bounds_unresolved_consumed_failure():
    messages = [
        _assistant("old"),
        _tool("old", "upstream failed\n" + ("x" * 8_000), status="error"),
        {"role": "assistant", "content": "I need another approach."},
    ]
    edited, report = edit_tool_context(messages, phase="failures")
    assert len(edited[1]["content"]) < 700
    assert "unresolved tool blocker" in edited[1]["content"]
    assert "upstream failed" in edited[1]["content"]
    assert report[0]["action"] == "blocker_receipt"


def test_consumed_large_read_is_saved_before_receipt(tmp_path):
    message = _tool("large", "e" * 16_001, status="success")
    message["_tool_receipt"]["consumed_turn"] = None
    messages = [_assistant("large"), message]
    mark_tool_results_consumed(
        messages,
        consumed_turn=7,
        artifact_dir=str(tmp_path),
        persist_artifacts=True,
    )
    receipt = messages[1]["_tool_receipt"]
    assert receipt["artifact_ref"]
    artifact = Path(receipt["artifact_ref"])
    assert artifact.read_text(encoding="utf-8") == "e" * 16_001
    assert artifact.stat().st_mode & 0o777 == 0o600


def test_tool_editor_does_not_receiptize_unconsumed_read_result():
    message = _tool("old", "not consumed")
    message["_tool_receipt"]["consumed_turn"] = None
    edited, report = edit_tool_context(
        [_assistant("old"), message, {"role": "assistant", "content": "done"}],
        phase="readonly",
    )
    assert edited[1]["content"] == "not consumed"
    assert report == []


def test_tool_editor_does_not_clear_unconsumed_empty_result():
    message = _tool("old", "", status="empty")
    message["_tool_receipt"]["consumed_turn"] = None
    edited, report = edit_tool_context(
        [_assistant("old"), message, {"role": "assistant", "content": "done"}],
        phase="active",
    )
    assert edited[1]["content"] == ""
    assert report == []


def test_tool_editor_recognizes_real_steer_marker():
    message = _tool("old", "duplicate" + format_steer_marker("stop now"))
    message["_tool_receipt"]["steer_present"] = False
    current = _tool("new", "duplicate" + format_steer_marker("stop now"))
    current["_tool_receipt"]["consumed_turn"] = None
    edited, report = edit_tool_context(
        [
            _assistant("old"),
            message,
            _assistant("new"),
            current,
        ],
        phase="active",
    )
    assert edited[1]["content"] == message["content"]
    assert report == []


def test_tool_editor_never_removes_assistant_reasoning_with_pair():
    assistant = _assistant("old")
    assistant["reasoning_content"] = "signed/provider reasoning"
    edited, report = edit_tool_context(
        [
            assistant,
            _tool("old", "", status="empty"),
            {"role": "assistant", "content": "done"},
        ],
        phase="active",
    )
    assert edited[0]["reasoning_content"] == "signed/provider reasoning"
    assert "body removed" in edited[1]["content"]
    assert report[0]["action"] == "placeholder"


def test_mark_consumed_persists_large_read_result_artifact(tmp_path):
    content = "evidence\n" * 2_000
    message = _tool("read-1", content)
    message["_tool_receipt"]["consumed_turn"] = None
    mark_tool_results_consumed(
        [_assistant("read-1"), message],
        consumed_turn=2,
        artifact_dir=str(tmp_path / "artifacts"),
        persist_artifacts=True,
    )
    receipt = message["_tool_receipt"]
    assert receipt["consumed_turn"] == 2
    assert receipt["artifact_ref"]
    artifact = Path(receipt["artifact_ref"])
    assert artifact.read_text(encoding="utf-8") == content
    assert artifact.stat().st_mode & 0o777 == 0o600


def test_mark_consumed_persists_smart_search_request_ledger_to_session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("research-session", source="subagent")
    assistant = _assistant(
        "fetch-db",
        "mcp__smart_search__smart_fetch",
        {"url": "https://example.test/db", "question": "What changed?"},
    )
    message = _tool("fetch-db", "pre-spilled preview")
    message["name"] = message["tool_name"] = "mcp__smart_search__smart_fetch"
    message["_tool_receipt"]["tool_name"] = "mcp__smart_search__smart_fetch"
    db.append_message(
        "research-session",
        "tool",
        "pre-spilled preview",
        tool_name="mcp__smart_search__smart_fetch",
        tool_call_id="fetch-db",
        effect_disposition="none",
        tool_receipt=message["_tool_receipt"],
    )

    mark_tool_results_consumed(
        [assistant, message],
        consumed_turn=3,
        session_db=db,
        session_id="research-session",
    )

    assert message["_tool_receipt"]["request_ledger"] == {
        "url": "https://example.test/db",
        "question": "What changed?",
    }
    loaded = db.get_messages_as_conversation("research-session")
    assert loaded[0]["_tool_receipt"]["request_ledger"] == {
        "url": "https://example.test/db",
        "question": "What changed?",
    }


def test_mark_consumed_does_not_persist_artifact_unless_enabled(tmp_path):
    content = "interactive evidence\n" * 400
    message = _tool("read-interactive", content)
    message["_tool_receipt"]["consumed_turn"] = None
    artifact_dir = tmp_path / "artifacts"

    mark_tool_results_consumed(
        [_assistant("read-interactive"), message],
        consumed_turn=2,
        artifact_dir=str(artifact_dir),
    )

    receipt = message["_tool_receipt"]
    assert receipt["consumed_turn"] == 2
    assert not receipt["artifact_ref"]
    assert not artifact_dir.exists()


def test_mark_consumed_updates_session_database_receipt(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session", source="cli")
    db.append_message(
        "session",
        "tool",
        "evidence",
        tool_name="web_search",
        tool_call_id="read-db",
        effect_disposition="none",
        tool_receipt=_tool("read-db", "evidence")["_tool_receipt"],
    )
    message = _tool("read-db", "evidence")
    message["_tool_receipt"]["consumed_turn"] = None
    mark_tool_results_consumed(
        [_assistant("read-db"), message],
        consumed_turn=4,
        session_db=db,
        session_id="session",
    )
    loaded = db.get_messages_as_conversation("session")
    assert loaded[0]["_tool_receipt"]["consumed_turn"] == 4


def test_smart_search_and_context7_mcp_tools_are_explicitly_read_only():
    assert not tool_may_have_side_effect("mcp__smart_search__smart_search")
    assert not tool_may_have_side_effect("mcp__smart_search__smart_fetch")
    assert not tool_may_have_side_effect("mcp__context7__query_docs")
    assert tool_may_have_side_effect("mcp__mixed_server__update_record")


def test_research_leaf_prompt_has_short_dedupe_and_partial_handoff_protocol():
    assert "运行时会自动" in RESEARCH_LEAF_PROMPT
    assert "不要自行创建、写入或选择 evidence 文件路径" in RESEARCH_LEAF_PROMPT
    assert "先广度覆盖任务中每个明确对象" in RESEARCH_LEAF_PROMPT
    assert "canonical 参数不得重复检索" in RESEARCH_LEAF_PROMPT
    assert "duplicate receipt" in RESEARCH_LEAF_PROMPT
    assert "预留最终 handoff" in RESEARCH_LEAF_PROMPT
    assert "max_iterations 只能标记 partial/unresolved" in RESEARCH_LEAF_PROMPT


def test_read_receipt_remains_provider_paired_across_wire_adapters():
    raw_body = "RSS evidence that must not replay"
    messages = [
        {"role": "user", "content": "research"},
        _assistant("rss-call"),
        _tool("rss-call", raw_body),
        {"role": "assistant", "content": "Evidence recorded."},
    ]
    edited, _ = edit_tool_context(messages, phase="readonly")
    strip_internal_tool_metadata(edited)

    from run_agent import AIAgent

    chat = AIAgent._sanitize_api_messages(edited)
    call_ids = {
        call["id"]
        for message in chat
        for call in (message.get("tool_calls") or [])
    }
    result_ids = {
        message["tool_call_id"]
        for message in chat
        if message.get("role") == "tool"
    }
    assert call_ids == result_ids == {"rss-call"}
    assert raw_body not in json.dumps(chat)

    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    responses = _chat_messages_to_responses_input(chat)
    response_calls = {
        item["call_id"]
        for item in responses
        if item.get("type") == "function_call"
    }
    response_outputs = {
        item["call_id"]
        for item in responses
        if item.get("type") == "function_call_output"
    }
    assert response_calls == response_outputs == {"rss-call"}

    from agent.anthropic_adapter import convert_messages_to_anthropic

    _, anthropic = convert_messages_to_anthropic(chat)
    tool_uses = {
        block["id"]
        for message in anthropic
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
        if block.get("type") == "tool_use"
    }
    tool_results = {
        block["tool_use_id"]
        for message in anthropic
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
        if block.get("type") == "tool_result"
    }
    assert tool_uses == tool_results == {"rss-call"}


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


@pytest.mark.parametrize(
    "tool_name",
    [
        "terminal",
        "mcp__github__search_code",
        "mcp__mixed_server__unknown_tool",
    ],
)
def test_tool_editor_failures_preserves_consumed_unknown_result(tool_name):
    body = "read result\n" + ("x" * 40_000)
    message = _tool("old", body)
    message["name"] = message["tool_name"] = tool_name
    message["_tool_receipt"].update(
        {"tool_name": tool_name, "effect": "unknown"}
    )
    edited, report = edit_tool_context(
        [
            _assistant("old", tool_name),
            message,
            {"role": "assistant", "content": "Evidence consumed."},
        ],
        phase="failures",
    )
    assert edited[1]["content"] == body
    assert report == []


def test_tool_editor_active_action_receipt_is_idempotent():
    message = _tool("old", "created external object")
    message["_tool_receipt"]["effect"] = "unknown"
    first, report = edit_tool_context(
        [
            _assistant("old", "send_message"),
            message,
            {"role": "assistant", "content": "done"},
        ],
        phase="active",
    )
    first_text = first[1]["content"]
    assert report[0]["action"] == "action_receipt"

    second, second_report = edit_tool_context(first, phase="active")
    assert second[1]["content"] == first_text
    assert second[1]["_tool_context_form"] == "action_receipt"
    assert second_report == []


def test_tool_editor_retains_roster_until_create_transition():
    roster = _tool(
        "roster",
        '{"profiles":[{"name":"research","description":"research"}]}',
    )
    roster["_tool_receipt"]["retain_until"] = "kanban_create"
    before_create = [
        _assistant("roster", "kanban_roster"),
        roster,
        {"role": "assistant", "content": "considering roster"},
    ]
    retained, report = edit_tool_context(before_create, phase="failures")
    assert retained[1]["content"] == roster["content"]
    assert report == []

    after_create = [
        *before_create,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "create",
                    "function": {
                        "name": "tool_call",
                        "arguments": json.dumps(
                            {
                                "name": "kanban_create",
                                "arguments": {
                                    "title": "research",
                                    "assignee": "research",
                                },
                            }
                        ),
                    },
                }
            ],
        },
    ]
    released, released_report = edit_tool_context(
        after_create,
        phase="failures",
    )
    assert "tool read result consumed" in released[1]["content"]
    assert released_report[0]["action"] == "read_receipt"


def test_tool_editor_bounds_unknown_effect_failure_as_blocker():
    message = _tool(
        "failed",
        "remote mutation status unknown\n" + ("z" * 20_000),
        status="error",
    )
    message["_tool_receipt"]["effect"] = "unknown"
    edited, report = edit_tool_context(
        [
            _assistant("failed", "mcp__mixed__write"),
            message,
            {"role": "assistant", "content": "Need operator decision."},
        ],
        phase="failures",
    )
    assert len(edited[1]["content"]) < 900
    assert "unresolved tool blocker" in edited[1]["content"]
    assert "remote mutation status unknown" in edited[1]["content"]
    assert report[0]["action"] == "blocker_receipt"


def test_tool_editor_failures_receiptizes_success_and_drops_superseded_failure():
    failed = _tool("failed", '{"error":"temporary"}', status="error")
    succeeded = _tool("succeeded", '{"ok":true,"content":"evidence"}')
    edited, report = edit_tool_context(
        [
            _assistant("failed", "mcp__smart_search__smart_fetch"),
            failed,
            _assistant("succeeded", "mcp__smart_search__smart_fetch"),
            succeeded,
            {"role": "assistant", "content": "Evidence used."},
        ],
        phase="failures",
    )

    assert all(message.get("tool_call_id") != "failed" for message in edited)
    success_message = next(
        message for message in edited
        if message.get("tool_call_id") == "succeeded"
    )
    assert "tool read result consumed" in success_message["content"]
    assert {item["action"] for item in report} == {"remove_pair", "read_receipt"}
    assert all(item["action"] != "action_receipt" for item in report)


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
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hello"}],
            "extra_headers": {"Authorization": "Bearer secret"},
        },
        request_id="req-1",
    )
    assert path is not None
    payload = json.loads(path.read_text())
    assert "extra_headers" not in payload["body"]
    display = json.loads(
        path.with_name("req-1.redacted.json").read_text(encoding="utf-8")
    )
    assert display["body"]["max_tokens"] == 4096
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
