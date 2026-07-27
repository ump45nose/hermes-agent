import json

from agent.capability_history import durable_projection


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
