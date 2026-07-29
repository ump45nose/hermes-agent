import json

from agent.capability_history import durable_projection, edit_consumed_transients


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_pure_search_describe_pairs_are_removed():
    messages = [
        {"role": "user", "content": "find a tool"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("s1", "tool_search", {"query": "files"}),
        ]},
        {"role": "tool", "tool_call_id": "s1", "content": "large index"},
        {"role": "assistant", "content": "done"},
    ]
    projected = durable_projection(messages)
    assert [m["role"] for m in projected] == ["user", "assistant"]
    assert "large index" not in json.dumps(projected)


def test_mixed_batch_removes_only_transient_call_id():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            _call("s1", "tool_describe", {"name": "terminal"}),
            _call("r1", "tool_call", {
                "name": "terminal",
                "arguments": {"command": "pwd"},
            }),
        ]},
        {"role": "tool", "tool_call_id": "s1", "content": "schema"},
        {
            "role": "tool",
            "tool_call_id": "r1",
            "tool_name": "terminal",
            "content": "/tmp",
        },
    ]
    projected = durable_projection(messages)
    assert len(projected) == 2
    calls = projected[0]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"
    assert calls[0]["resolved_via"] == "tool_call"
    assert projected[1]["content"] == "/tmp"


def test_skill_view_body_is_transient():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            _call("k1", "tool_call", {
                "name": "skill_view",
                "arguments": {"name": "example"},
            }),
        ]},
        {"role": "tool", "tool_call_id": "k1", "content": "FULL SKILL BODY"},
    ]
    assert durable_projection(messages) == []


def test_consumed_disclosure_is_visible_once_then_removed():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            _call("s1", "tool_search", {"query": "web"}),
        ]},
        {
            "role": "tool",
            "tool_call_id": "s1",
            "tool_name": "tool_search",
            "content": "large discovery payload",
            "_tool_receipt": {"consumed_turn": None},
        },
    ]
    first, report = edit_consumed_transients(messages)
    assert first == messages
    assert report == []

    messages[1]["_tool_receipt"]["consumed_turn"] = 2
    second, report = edit_consumed_transients(messages)
    assert second == []
    assert {row["action"] for row in report} == {"remove_call", "remove_result"}


def test_consumed_mixed_batch_removes_only_disclosure():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            _call("s1", "tool_describe", {"name": "web_search"}),
            _call("r1", "tool_call", {
                "name": "web_search",
                "arguments": {"query": "Hermes"},
            }),
        ]},
        {
            "role": "tool",
            "tool_call_id": "s1",
            "tool_name": "tool_describe",
            "content": "schema",
            "_tool_receipt": {"consumed_turn": 3},
        },
        {
            "role": "tool",
            "tool_call_id": "r1",
            "tool_name": "web_search",
            "content": "result",
            "_tool_receipt": {"consumed_turn": None},
        },
    ]
    edited, _ = edit_consumed_transients(messages)
    assert len(edited) == 2
    assert [c["id"] for c in edited[0]["tool_calls"]] == ["r1"]
    assert edited[1]["tool_call_id"] == "r1"


def test_consumed_signed_assistant_keeps_minimum_pair():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "provider-signed reasoning",
            "tool_calls": [_call("s1", "skill_search", {"query": "docs"})],
        },
        {
            "role": "tool",
            "tool_call_id": "s1",
            "tool_name": "skill_search",
            "content": "large skill index",
            "_tool_receipt": {"consumed_turn": 4},
        },
    ]
    edited, report = edit_consumed_transients(messages)
    assert len(edited) == 2
    assert edited[0]["tool_calls"][0]["id"] == "s1"
    assert edited[1]["content"] == (
        "[capability disclosure consumed; payload removed]"
    )
    assert report == [{"tool_call_id": "s1", "action": "placeholder"}]
