import json
from types import SimpleNamespace

from agent.controller_protocol import (
    CONTROLLER_DISPATCH_ACK,
    CONTROLLER_PROTOCOL,
    CONTROLLER_RECEIPT_PREFIX,
    controller_create_call_ids,
    controller_dispatch_outcome,
    controller_tool_policy_block,
    build_controller_dispatch_nudge,
    encode_controller_receipt,
    note_controller_tool_result,
    runtime_protocol_tool_policy_block,
)
from gateway.kanban_watchers import _controller_receipt_for_event


def _call(call_id: str, name: str = "kanban_create"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name),
    )


def _bridge_call(call_id: str, target: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="tool_call",
            arguments=json.dumps({"name": target, "arguments": {}}),
        ),
    )


def _agent():
    return SimpleNamespace(_prompt_lock={"protocols": [CONTROLLER_PROTOCOL]})


def test_controller_only_parks_a_pure_successful_create_batch():
    assistant = SimpleNamespace(tool_calls=[_call("c1"), _call("c2")])
    call_ids = controller_create_call_ids(_agent(), assistant)
    messages = [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {
                    "ok": True,
                    "task_id": task_id,
                    "subscribed": True,
                    "controller_batch_id": "batch-1",
                }
            ),
        }
        for call_id, task_id in (("c1", "task-1"), ("c2", "task-2"))
    ]

    outcome = controller_dispatch_outcome(call_ids, messages)
    assert outcome.parked
    assert outcome.task_ids == ("task-1", "task-2")
    assert outcome.batch_id == "batch-1"
    assert CONTROLLER_DISPATCH_ACK == "已派单，等待执行结果。"


def test_controller_does_not_park_on_failed_or_mixed_tool_batch():
    mixed = SimpleNamespace(
        tool_calls=[_call("c1"), _call("c2", "kanban_roster")]
    )
    assert controller_create_call_ids(_agent(), mixed) == []

    failed = controller_dispatch_outcome(
        ["c1"],
        [
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": json.dumps(
                    {
                        "ok": True,
                        "task_id": "task-1",
                        "subscribed": False,
                        "controller_batch_id": "batch-1",
                    }
                ),
            }
        ],
    )
    assert not failed.parked
    assert "subscription" in failed.reason


def test_controller_recognizes_kanban_create_through_progressive_bridge():
    assistant = SimpleNamespace(
        tool_calls=[
            _bridge_call("c1", "kanban_create"),
            _bridge_call("c2", "kanban_create"),
        ]
    )
    assert controller_create_call_ids(_agent(), assistant) == ["c1", "c2"]


def test_controller_roster_selection_requires_create_before_text_exit():
    agent = _agent()
    agent.runtime_role = "interactive"
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "r1",
                    "function": {
                        "name": "tool_call",
                        "arguments": json.dumps(
                            {"name": "kanban_roster", "arguments": {}}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_roster",
            "tool_call_id": "r1",
            "content": '{"profiles": []}',
        },
    ]

    nudge = build_controller_dispatch_nudge(
        agent,
        messages=messages,
    )
    assert nudge is not None
    assert "kanban_create" in nudge
    assert "triage=true" in nudge
    assert (
        build_controller_dispatch_nudge(
            agent,
            messages=messages,
            attempts=2,
        )
        is None
    )


def test_controller_dispatch_nudge_stops_after_create_attempt():
    agent = _agent()
    agent.runtime_role = "interactive"
    messages = [
        {
            "role": "assistant",
            "tool_calls": [_call("r1", "kanban_roster")],
        },
        {
            "role": "assistant",
            "tool_calls": [_bridge_call("c1", "kanban_create")],
        },
    ]
    assert (
        build_controller_dispatch_nudge(
            agent,
            messages=messages,
        )
        is None
    )


def test_controller_repeated_roster_is_blocked_after_success():
    agent = _agent()
    agent.runtime_role = "interactive"
    note_controller_tool_result(agent, "kanban_roster", "success")

    reason = controller_tool_policy_block(agent, "kanban_roster")
    assert reason is not None
    assert "kanban_create" in reason
    protocol, next_tool, _ = runtime_protocol_tool_policy_block(
        agent, "kanban_roster"
    )
    assert protocol == CONTROLLER_PROTOCOL
    assert next_tool == "kanban_create"
    assert controller_tool_policy_block(agent, "kanban_create") is None

    note_controller_tool_result(agent, "kanban_create", "error")
    assert controller_tool_policy_block(agent, "kanban_roster") is None


def test_controller_specialist_action_boundary_is_static_and_role_scoped():
    agent = _agent()
    agent.runtime_role = "interactive"
    for tool_name in (
        "terminal",
        "web_search",
        "mcp__github__get_file_contents",
        "mcp__smart_search__smart_research",
        "mcp__shared_state__get_resource",
    ):
        reason = controller_tool_policy_block(agent, tool_name)
        assert reason is not None
        assert "kanban_roster" in reason

    assert controller_tool_policy_block(agent, "kanban_roster") is None
    assert controller_tool_policy_block(agent, "kanban_create") is None
    assert controller_tool_policy_block(agent, "mcp__memos__search_memos") is None

    agent.runtime_role = "cron"
    assert controller_tool_policy_block(agent, "terminal") is None


def test_research_parent_must_delegate_source_access_to_leaf():
    agent = SimpleNamespace(
        _prompt_lock={"protocols": ["research-parent@1"]},
        runtime_role="kanban_worker",
    )
    for tool_name in (
        "web_search",
        "browser_navigate",
        "mcp__github__get_file_contents",
        "mcp__smart_search__smart_research",
    ):
        protocol, next_tool, reason = runtime_protocol_tool_policy_block(
            agent, tool_name
        )
        assert protocol == "research-parent@1"
        assert next_tool == "delegate_task"
        assert "Only research_leaf" in reason

    assert runtime_protocol_tool_policy_block(agent, "delegate_task") is None
    assert runtime_protocol_tool_policy_block(agent, "kanban_complete") is None
    agent.runtime_role = "research_leaf"
    assert (
        runtime_protocol_tool_policy_block(
            agent, "mcp__github__get_file_contents"
        )
        is None
    )


def test_controller_receipt_uses_live_batch_barrier_and_internal_prefix():
    task = SimpleNamespace(
        id="task-1",
        status="done",
        assignee="research",
        result="fallback",
        controller_batch_id="batch-1",
    )
    sibling = SimpleNamespace(
        id="task-2",
        status="running",
        assignee="ops",
        controller_batch_id="batch-1",
    )
    event = SimpleNamespace(
        kind="completed",
        payload={
            "summary": "evidence ready",
            "artifacts": ["/artifact/report.md"],
            "unresolved": [],
        },
    )

    receipt = _controller_receipt_for_event(
        task=task,
        event=event,
        batch_tasks=[task, sibling],
    )
    assert receipt == {
        "protocol": "kanban-controller@1",
        "task_id": "task-1",
        "status": "done",
        "assignee": "research",
        "summary": "evidence ready",
        "artifacts": ["/artifact/report.md"],
        "unresolved": [],
        "retry_expected": False,
        "remaining_tasks": [
            {
                "task_id": "task-2",
                "status": "running",
                "assignee": "ops",
            }
        ],
        "controller_batch_id": "batch-1",
    }
    encoded = encode_controller_receipt(receipt)
    assert encoded.startswith(CONTROLLER_RECEIPT_PREFIX + "\n")
    assert json.loads(encoded.split("\n", 1)[1]) == receipt
