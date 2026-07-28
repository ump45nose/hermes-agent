"""Deterministic Controller dispatch and receipt helpers."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any


CONTROLLER_PROTOCOL = "kanban-controller@1"
RESEARCH_PARENT_PROTOCOL = "research-parent@1"
CONTROLLER_RECEIPT_PREFIX = "[[HERMES_CONTROLLER_RECEIPT_V1]]"
CONTROLLER_DISPATCH_ACK = "已派单，等待执行结果。"

_DELEGATED_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "read_file",
        "write_file",
        "patch",
        "search_files",
        "web_search",
        "web_extract",
        "delegate_task",
    }
)
_DELEGATED_TOOL_PREFIXES = (
    "browser_",
    "mcp__github__",
    "mcp__smart_search__",
    "mcp__smartsearch_remote__",
    "mcp__context7__",
    "mcp__shared_state__",
)

_RESEARCH_LEAF_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "read_file",
        "write_file",
        "patch",
        "search_files",
        "web_search",
        "web_extract",
    }
)
_RESEARCH_LEAF_TOOL_PREFIXES = (
    "browser_",
    "mcp__github__",
    "mcp__smart_search__",
    "mcp__smartsearch_remote__",
    "mcp__context7__",
)


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
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments")
    else:
        name = str(getattr(function, "name", "") or "")
        raw_arguments = getattr(function, "arguments", None)
    if name != "tool_call":
        return name
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except (TypeError, ValueError):
            return name
    if isinstance(raw_arguments, dict):
        return str(raw_arguments.get("name") or name)
    return name


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


def controller_tool_policy_block(
    agent: Any,
    tool_name: str,
) -> str | None:
    """Keep specialist execution outside an interactive Controller process.

    This is an action boundary, not semantic query routing. The model remains
    responsible for deciding direct handling, assignee, decomposition, and
    acceptance, but it cannot use broad specialist tools as an escape hatch
    from the declared Kanban protocol.
    """
    if not controller_protocol_enabled(agent):
        return None
    if str(getattr(agent, "runtime_role", "interactive")) != "interactive":
        return None
    name = str(tool_name or "")
    delegated = (
        name in _DELEGATED_TOOL_NAMES
        or any(name.startswith(prefix) for prefix in _DELEGATED_TOOL_PREFIXES)
    )
    if not delegated:
        return None
    return (
        f"Controller protocol blocked specialist tool {name!r}. "
        "Call kanban_roster, then kanban_create with a current assignee; "
        "use triage=true when the task is ambiguous or crosses domains. "
        "Tool visibility does not make specialist execution part of the "
        "Controller's direct responsibility."
    )


def runtime_protocol_tool_policy_block(
    agent: Any,
    tool_name: str,
) -> tuple[str, str, str] | None:
    """Return ``(protocol, required_next_tool, error)`` for a hard boundary."""
    controller_error = controller_tool_policy_block(agent, tool_name)
    if controller_error is not None:
        return CONTROLLER_PROTOCOL, "kanban_roster", controller_error

    protocols = set((getattr(agent, "_prompt_lock", None) or {}).get("protocols") or [])
    if RESEARCH_PARENT_PROTOCOL not in protocols:
        return None
    if str(getattr(agent, "runtime_role", "")) == "research_leaf":
        return None
    name = str(tool_name or "")
    if not (
        name in _RESEARCH_LEAF_TOOL_NAMES
        or any(name.startswith(prefix) for prefix in _RESEARCH_LEAF_TOOL_PREFIXES)
    ):
        return None
    return (
        RESEARCH_PARENT_PROTOCOL,
        "delegate_task",
        (
            f"Research parent protocol blocked leaf source tool {name!r}. "
            "Split the research goal into three complementary tasks and call "
            "delegate_task synchronously. Only research_leaf processes may "
            "search/fetch sources; the parent waits for every terminal handoff, "
            "then synthesizes or inspects a scoped artifact when evidence is missing."
        ),
    )


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
