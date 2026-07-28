"""Deterministic Controller dispatch and receipt helpers."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any


CONTROLLER_PROTOCOL = "kanban-controller@1"
CONTROLLER_RECEIPT_PREFIX = "[[HERMES_CONTROLLER_RECEIPT_V1]]"
CONTROLLER_DISPATCH_ACK = "已派单，等待执行结果。"


def controller_protocol_enabled(agent: Any) -> bool:
    return CONTROLLER_PROTOCOL in set(
        (getattr(agent, "_prompt_lock", {}) or {}).get("protocols") or []
    )


def profile_controller_protocol_enabled(profile: str) -> bool:
    try:
        from hermes_constants import get_canonical_hermes_root, get_profile_home
        from hermes_cli.prompt_compiler import load_compiled_prompt

        loaded = load_compiled_prompt(
            get_profile_home(
                str(profile or "default"),
                root=get_canonical_hermes_root(),
            )
        )
    except Exception:
        return False
    return bool(
        loaded
        and CONTROLLER_PROTOCOL in set((loaded[1].get("protocols") or []))
    )


def new_controller_batch_id(agent: Any) -> str:
    session = str(getattr(agent, "session_id", "") or "session")
    return f"cb_{session[:12]}_{uuid.uuid4().hex[:12]}"


def _call_name(call: Any) -> str:
    function = (
        call.get("function")
        if isinstance(call, dict)
        else getattr(call, "function", None)
    )
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(getattr(function, "name", "") or "")


def _call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or call.get("call_id") or "")
    return str(getattr(call, "id", "") or getattr(call, "call_id", "") or "")


def controller_create_call_ids(agent: Any, assistant_message: Any) -> list[str]:
    if not controller_protocol_enabled(agent):
        return []
    calls = list(getattr(assistant_message, "tool_calls", None) or [])
    if not calls or any(_call_name(call) != "kanban_create" for call in calls):
        return []
    return [_call_id(call) for call in calls if _call_id(call)]


@dataclass(frozen=True)
class ControllerDispatchOutcome:
    parked: bool
    batch_id: str = ""
    task_ids: tuple[str, ...] = ()
    reason: str = ""


def controller_dispatch_outcome(
    call_ids: list[str],
    messages: list[dict[str, Any]],
) -> ControllerDispatchOutcome:
    """Return parked=True only when every create landed and subscribed."""
    if not call_ids:
        return ControllerDispatchOutcome(False, reason="not a controller create batch")
    by_id = {
        str(message.get("tool_call_id") or ""): message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    }
    task_ids: list[str] = []
    batch_ids: set[str] = set()
    for call_id in call_ids:
        message = by_id.get(call_id)
        if message is None:
            return ControllerDispatchOutcome(False, reason=f"missing result {call_id}")
        try:
            payload = json.loads(str(message.get("content") or ""))
        except (TypeError, ValueError):
            return ControllerDispatchOutcome(False, reason=f"invalid result {call_id}")
        if not isinstance(payload, dict) or not payload.get("ok"):
            return ControllerDispatchOutcome(False, reason=f"create failed {call_id}")
        if not payload.get("subscribed"):
            return ControllerDispatchOutcome(
                False, reason=f"subscription unavailable {call_id}"
            )
        task_id = str(payload.get("task_id") or "")
        batch_id = str(payload.get("controller_batch_id") or "")
        if not task_id or not batch_id:
            return ControllerDispatchOutcome(
                False, reason=f"missing controller identity {call_id}"
            )
        task_ids.append(task_id)
        batch_ids.add(batch_id)
    if len(batch_ids) != 1:
        return ControllerDispatchOutcome(False, reason="mixed controller batch ids")
    return ControllerDispatchOutcome(
        True,
        batch_id=next(iter(batch_ids)),
        task_ids=tuple(task_ids),
    )


def encode_controller_receipt(receipt: dict[str, Any]) -> str:
    return CONTROLLER_RECEIPT_PREFIX + "\n" + json.dumps(
        receipt, ensure_ascii=False, sort_keys=True
    )


def is_controller_receipt_text(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(CONTROLLER_RECEIPT_PREFIX)
