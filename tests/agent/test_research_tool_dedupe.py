import json
import threading
from concurrent.futures import ThreadPoolExecutor

import model_tools
from agent.research_tool_dedupe import ResearchToolDeduper


TOOL = "mcp__smart_search__smart_fetch"


def _call(monkeypatch, dispatch, *, session, role="research_leaf", arguments=None):
    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    return model_tools.handle_function_call(
        TOOL,
        arguments or {"url": "https://example.test/article"},
        session_id=session,
        runtime_role=role,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )


def test_research_leaf_sequential_exact_duplicate_skips_second_dispatch(monkeypatch):
    calls = []

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        return json.dumps({"ok": True, "content": "evidence"})

    first = _call(monkeypatch, dispatch, session="dedupe-sequential")
    second = _call(monkeypatch, dispatch, session="dedupe-sequential")

    assert json.loads(first)["content"] == "evidence"
    receipt = json.loads(second)
    assert len(calls) == 1
    assert receipt["kind"] == "smart_search_duplicate_receipt"
    assert receipt["duplicate"] is True
    assert receipt["request"] == {"url": "https://example.test/article"}
    assert "content" not in receipt["original_result"]


def test_research_leaf_parallel_exact_duplicate_is_single_flight(monkeypatch):
    calls = []
    entered = threading.Event()
    release = threading.Event()

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(timeout=5)
        return json.dumps({"ok": True, "content": "parallel evidence"})

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    arguments = {"query": "parallel exact query"}
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            model_tools.handle_function_call,
            TOOL,
            arguments,
            session_id="dedupe-parallel",
            runtime_role="research_leaf",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
        assert entered.wait(timeout=5)
        second = executor.submit(
            model_tools.handle_function_call,
            TOOL,
            dict(arguments),
            session_id="dedupe-parallel",
            runtime_role="research_leaf",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
        release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert len(calls) == 1
    decoded = [json.loads(result) for result in results]
    assert sum(item.get("duplicate") is True for item in decoded) == 1
    assert sum(item.get("content") == "parallel evidence" for item in decoded) == 1


def test_parallel_duplicate_wait_is_bounded_when_owner_hangs():
    deduper = ResearchToolDeduper(single_flight_wait_seconds=0.01)
    arguments = {"query": "owner still running"}
    owner = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-hung-owner",
        tool_name=TOOL,
        arguments=arguments,
    )

    duplicate = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-hung-owner",
        tool_name=TOOL,
        arguments=dict(arguments),
    )

    assert owner.owner is True
    receipt = json.loads(duplicate.duplicate_result)
    assert receipt["ok"] is False
    assert receipt["kind"] == "smart_search_inflight_timeout_receipt"
    assert receipt["status"] == "timeout"
    assert receipt["request"] == arguments
    deduper.abort(owner)


def test_dedupe_isolated_by_session_and_disabled_for_interactive_parent(monkeypatch):
    calls = []

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        return json.dumps({"ok": True, "content": f"call-{len(calls)}"})

    _call(monkeypatch, dispatch, session="dedupe-isolation-a")
    _call(monkeypatch, dispatch, session="dedupe-isolation-b")
    _call(monkeypatch, dispatch, session="dedupe-interactive", role="interactive")
    _call(monkeypatch, dispatch, session="dedupe-interactive", role="interactive")

    assert len(calls) == 4


def test_failed_call_is_not_cached_as_success(monkeypatch):
    results = [
        json.dumps({"ok": False, "error": "temporary failure"}),
        json.dumps({"ok": True, "content": "recovered"}),
    ]
    calls = []

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        return results.pop(0)

    first = _call(monkeypatch, dispatch, session="dedupe-failure")
    second = _call(monkeypatch, dispatch, session="dedupe-failure")
    third = _call(monkeypatch, dispatch, session="dedupe-failure")

    assert json.loads(first)["ok"] is False
    assert json.loads(second)["content"] == "recovered"
    assert json.loads(third)["duplicate"] is True
    assert len(calls) == 2


def test_nested_mcp_failure_is_not_cached_as_success(monkeypatch):
    results = [
        json.dumps(
            {
                "result": json.dumps(
                    {"ok": False, "error": "upstream unavailable"}
                )
            }
        ),
        json.dumps({"result": json.dumps({"ok": True, "content": "recovered"})}),
    ]
    calls = []

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        return results.pop(0)

    first = _call(monkeypatch, dispatch, session="dedupe-nested-failure")
    second = _call(monkeypatch, dispatch, session="dedupe-nested-failure")
    third = _call(monkeypatch, dispatch, session="dedupe-nested-failure")

    assert json.loads(json.loads(first)["result"])["ok"] is False
    assert json.loads(json.loads(second)["result"])["ok"] is True
    assert json.loads(third)["duplicate"] is True
    assert len(calls) == 2


def test_duplicate_receipt_is_bounded_and_omits_unrelated_arguments(monkeypatch):
    def dispatch(*_args, **_kwargs):
        return json.dumps({"ok": True, "content": "x" * 50_000})

    arguments = {
        "question": "q" * 5_000,
        "url": "https://example.test/large",
        "irrelevant": "must-not-enter-receipt",
    }
    _call(
        monkeypatch,
        dispatch,
        session="dedupe-bounded-receipt",
        arguments=arguments,
    )
    receipt = _call(
        monkeypatch,
        dispatch,
        session="dedupe-bounded-receipt",
        arguments=arguments,
    )

    assert len(receipt) < 2_000
    assert "must-not-enter-receipt" not in receipt
    assert "x" * 100 not in receipt
    assert json.loads(receipt)["request"]["question"].endswith("…")
