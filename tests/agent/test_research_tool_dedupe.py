import json
import threading
from concurrent.futures import ThreadPoolExecutor

import model_tools
import pytest
from agent import research_tool_dedupe
from agent.research_tool_dedupe import ResearchToolDeduper
from tools.registry import registry


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


def test_hung_owner_lease_allows_one_takeover_and_later_success():
    deduper = ResearchToolDeduper(single_flight_wait_seconds=0.01)
    arguments = {"query": "owner still running"}
    owner = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-hung-owner",
        tool_name=TOOL,
        arguments=arguments,
    )

    takeover = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-hung-owner",
        tool_name=TOOL,
        arguments=dict(arguments),
    )

    assert owner.owner is True
    assert takeover.owner is True
    assert takeover.flight is not owner.flight
    assert owner.flight.event.is_set()

    # A late completion from the stale owner cannot overwrite the takeover.
    deduper.finish(owner, json.dumps({"ok": True, "content": "stale"}), arguments)
    deduper.finish(
        takeover,
        json.dumps({"ok": True, "content": "fresh"}),
        arguments,
    )
    duplicate = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-hung-owner",
        tool_name=TOOL,
        arguments=dict(arguments),
    )
    receipt = json.loads(duplicate.duplicate_result)
    assert receipt["kind"] == "smart_search_duplicate_receipt"
    assert receipt["original_result"]["status"] == "success"


def test_finish_exception_aborts_and_wakes_future_owner(monkeypatch):
    deduper = ResearchToolDeduper()
    arguments = {"query": "finish exception"}
    owner = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-finish-exception",
        tool_name=TOOL,
        arguments=arguments,
    )

    monkeypatch.setattr(
        research_tool_dedupe,
        "request_ledger",
        lambda _arguments: (_ for _ in ()).throw(RuntimeError("ledger failed")),
    )
    with pytest.raises(RuntimeError, match="ledger failed"):
        deduper.finish(
            owner,
            json.dumps({"ok": True, "content": "not cached"}),
            arguments,
        )

    assert owner.flight.event.is_set()
    replacement = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-finish-exception",
        tool_name=TOOL,
        arguments=arguments,
    )
    assert replacement.owner is True
    deduper.abort(replacement)


def test_success_cache_and_inflight_capacity_are_bounded():
    deduper = ResearchToolDeduper(
        max_entries=1,
        max_entries_per_session=1,
        max_inflight=1,
        single_flight_lease_seconds=1,
    )
    first_args = {"query": "first"}
    first = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-eviction",
        tool_name=TOOL,
        arguments=first_args,
    )
    deduper.finish(first, json.dumps({"ok": True}), first_args)

    second_args = {"query": "second"}
    second = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-eviction",
        tool_name=TOOL,
        arguments=second_args,
    )
    deduper.finish(second, json.dumps({"ok": True}), second_args)
    evicted_first = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-eviction",
        tool_name=TOOL,
        arguments=first_args,
    )
    assert evicted_first.owner is True

    at_capacity = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-eviction",
        tool_name=TOOL,
        arguments={"query": "third"},
    )
    assert at_capacity.owner is False
    assert at_capacity.duplicate_result is None
    assert len(deduper._inflight) == 1

    # Stale inflight ownership is evicted before the capacity check.
    evicted_first.flight.started_at -= 2
    replacement = deduper.begin(
        runtime_role="research_leaf",
        session_id="dedupe-eviction",
        tool_name=TOOL,
        arguments={"query": "third"},
    )
    assert replacement.owner is True
    assert evicted_first.flight.event.is_set()
    deduper.abort(replacement)


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


def test_execution_middleware_raw_success_to_error_is_not_cached(monkeypatch):
    calls = []

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        return json.dumps({"ok": True, "content": "raw success"})

    def rewrite_after_dispatch(**kwargs):
        kwargs["next_call"](kwargs["args"])
        return json.dumps({"ok": False, "error": "middleware rejected"})

    manager = type(
        "Manager",
        (),
        {"_middleware": {"tool_execution": [rewrite_after_dispatch]}},
    )()
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: manager,
    )

    first = _call(
        monkeypatch,
        dispatch,
        session="dedupe-middleware-final-error",
    )
    second = _call(
        monkeypatch,
        dispatch,
        session="dedupe-middleware-final-error",
    )

    assert json.loads(first)["ok"] is False
    assert json.loads(second)["ok"] is False
    assert len(calls) == 2


def test_execution_middleware_same_args_retry_does_not_wait_on_own_flight(
    monkeypatch,
):
    calls = []

    def dispatch(_name, args, **_kwargs):
        calls.append(dict(args))
        return json.dumps({"ok": True, "content": f"call-{len(calls)}"})

    def retry_twice(_tool_name, args, next_call, **_context):
        next_call(dict(args))
        return next_call(dict(args))

    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        retry_twice,
    )
    first = _call(
        monkeypatch,
        dispatch,
        session="dedupe-middleware-same-retry",
        arguments={"query": "same retry"},
    )
    duplicate = _call(
        monkeypatch,
        dispatch,
        session="dedupe-middleware-same-retry",
        arguments={"query": "same retry"},
    )

    assert json.loads(first)["content"] == "call-2"
    assert json.loads(duplicate)["duplicate"] is True
    assert calls == [{"query": "same retry"}, {"query": "same retry"}]


def test_execution_middleware_different_args_retry_caches_only_last_key(
    monkeypatch,
):
    calls = []

    def dispatch(_name, args, **_kwargs):
        calls.append(dict(args))
        return json.dumps({"ok": True, "query": args["query"]})

    def retry_different(_tool_name, _args, next_call, **_context):
        next_call({"query": "first attempt"})
        return next_call({"query": "last attempt"})

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        retry_different,
    )
    result = model_tools.handle_function_call(
        TOOL,
        {"query": "original"},
        session_id="dedupe-middleware-different-retry",
        runtime_role="research_leaf",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )
    assert json.loads(result)["query"] == "last attempt"

    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        lambda _tool_name, args, next_call, **_context: next_call(args),
    )
    first_again = model_tools.handle_function_call(
        TOOL,
        {"query": "first attempt"},
        session_id="dedupe-middleware-different-retry",
        runtime_role="research_leaf",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )
    last_again = model_tools.handle_function_call(
        TOOL,
        {"query": "last attempt"},
        session_id="dedupe-middleware-different-retry",
        runtime_role="research_leaf",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )

    assert json.loads(first_again)["query"] == "first attempt"
    assert json.loads(last_again)["duplicate"] is True
    assert calls == [
        {"query": "first attempt"},
        {"query": "last attempt"},
        {"query": "first attempt"},
    ]


def test_transform_raw_success_to_error_is_not_cached(monkeypatch):
    calls = []

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        return json.dumps({"ok": True, "content": "raw success"})

    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "transform_tool_result",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **_kwargs: (
            [json.dumps({"ok": False, "error": "transform rejected"})]
            if hook_name == "transform_tool_result"
            else []
        ),
    )

    first = _call(
        monkeypatch,
        dispatch,
        session="dedupe-transform-final-error",
    )
    second = _call(
        monkeypatch,
        dispatch,
        session="dedupe-transform-final-error",
    )

    assert json.loads(first)["ok"] is False
    assert json.loads(second)["ok"] is False
    assert len(calls) == 2


def test_model_tools_finish_exception_is_fail_open_and_not_cached(monkeypatch):
    calls = []

    def dispatch(*_args, **_kwargs):
        calls.append(1)
        return json.dumps({"ok": True, "content": "visible result"})

    monkeypatch.setattr(
        research_tool_dedupe,
        "finish_research_leaf_smart_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("finish failed")
        ),
    )

    first = _call(monkeypatch, dispatch, session="dedupe-finish-fail-open")
    second = _call(monkeypatch, dispatch, session="dedupe-finish-fail-open")

    assert json.loads(first)["content"] == "visible result"
    assert json.loads(second)["content"] == "visible result"
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


def test_registry_generation_refreshes_same_session_and_arguments():
    tool_name = "mcp__smart_search__dedupe_generation_test"
    schema = {
        "name": tool_name,
        "description": "Test dynamic SmartSearch entry.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    }
    calls = []

    def register_version(version):
        registry.register(
            name=tool_name,
            toolset="mcp-smart-search-test",
            schema=schema,
            handler=lambda _args, **_kwargs: (
                calls.append(version)
                or json.dumps({"ok": True, "version": version})
            ),
        )

    registry.deregister(tool_name)
    try:
        register_version(1)
        first = model_tools.handle_function_call(
            tool_name,
            {"query": "same"},
            session_id="dedupe-registry-generation",
            runtime_role="research_leaf",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
        duplicate = model_tools.handle_function_call(
            tool_name,
            {"query": "same"},
            session_id="dedupe-registry-generation",
            runtime_role="research_leaf",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
        assert json.loads(first)["version"] == 1
        assert json.loads(duplicate)["duplicate"] is True

        registry.deregister(tool_name)
        register_version(2)
        refreshed = model_tools.handle_function_call(
            tool_name,
            {"query": "same"},
            session_id="dedupe-registry-generation",
            runtime_role="research_leaf",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
        assert json.loads(refreshed)["version"] == 2
        assert calls == [1, 2]
    finally:
        registry.deregister(tool_name)
