import json
from types import SimpleNamespace

from agent.controller_protocol import (
    CONTROLLER_DISPATCH_ACK,
    CONTROLLER_PROTOCOL,
    CONTROLLER_RECEIPT_PREFIX,
    controller_create_call_ids,
    controller_dispatch_outcome,
    encode_controller_receipt,
)
from gateway.kanban_watchers import _controller_receipt_for_event


def _call(call_id: str, name: str = "kanban_create"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name),
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
