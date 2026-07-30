"""Durable projection for progressive capability-disclosure messages.

Tool-search references are durable session state because they determine which
real schemas are hydrated after resume. Legacy describe payloads and Skill
discovery remain transient scaffolding.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_TRANSIENT_BRIDGES = {"tool_describe", "skill_search"}
_TRANSIENT_UNDERLYING = {"skills_list", "skill_view"}


def _call_parts(call: Any) -> Tuple[str, str, Any]:
    """Return ``(call_id, name, arguments)`` for dict or SDK call objects."""
    if isinstance(call, dict):
        call_id = str(call.get("id") or call.get("call_id") or "")
        fn = call.get("function")
        if isinstance(fn, dict):
            return call_id, str(fn.get("name") or ""), fn.get("arguments")
        return call_id, str(call.get("name") or ""), call.get("arguments")
    fn = getattr(call, "function", None)
    return (
        str(getattr(call, "id", "") or getattr(call, "call_id", "") or ""),
        str(getattr(fn, "name", "") or ""),
        getattr(fn, "arguments", None),
    )


def _arguments_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def _is_transient_call(name: str, raw_arguments: Any) -> bool:
    if name in _TRANSIENT_BRIDGES:
        return True
    if name != "tool_call":
        return False
    underlying = str(_arguments_object(raw_arguments).get("name") or "")
    return underlying in _TRANSIENT_UNDERLYING


def transient_call_ids(messages: Iterable[Dict[str, Any]]) -> Set[str]:
    ids: Set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        calls = msg.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            call_id, name, arguments = _call_parts(call)
            if call_id and _is_transient_call(name, arguments):
                ids.add(call_id)
    return ids


def _receipt_consumed(msg: Dict[str, Any]) -> bool:
    raw = msg.get("_tool_receipt")
    return isinstance(raw, dict) and raw.get("consumed_turn") is not None


def _described_tool_name(
    messages: List[Dict[str, Any]],
    *,
    call_id: str,
) -> str:
    """Resolve the tool named by a ``tool_describe`` call/result pair."""
    for msg in messages:
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not isinstance(calls, list):
            continue
        for call in calls:
            candidate_id, name, raw_arguments = _call_parts(call)
            if candidate_id != call_id or name != "tool_describe":
                continue
            described = str(_arguments_object(raw_arguments).get("name") or "")
            if described:
                return described
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and str(msg.get("tool_call_id") or "") == call_id
        ):
            try:
                payload = json.loads(str(msg.get("content") or ""))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                return str(payload.get("name") or "")
    return ""


def _described_contract_pending(
    messages: List[Dict[str, Any]],
    *,
    describe_call_id: str,
) -> bool:
    """Keep a schema until the described tool result reaches the provider."""
    described = _described_tool_name(messages, call_id=describe_call_id)
    if not described:
        return False

    describe_call_index = -1
    describe_index = -1
    for index, msg in enumerate(messages):
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if isinstance(calls, list) and any(
            _call_parts(call)[0] == describe_call_id for call in calls
        ):
            describe_call_index = index
        if (
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and str(msg.get("tool_call_id") or "") == describe_call_id
        ):
            describe_index = index
            break
    if describe_index < 0:
        return False

    underlying_ids: Set[str] = set()
    scan_from = describe_call_index if describe_call_index >= 0 else describe_index + 1
    for msg in messages[scan_from:]:
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not isinstance(calls, list):
            continue
        for call in calls:
            candidate_id, name, raw_arguments = _call_parts(call)
            if candidate_id == describe_call_id:
                continue
            if name == "tool_call":
                name = str(_arguments_object(raw_arguments).get("name") or "")
            if candidate_id and name == described:
                underlying_ids.add(candidate_id)

    # The schema remains current-turn working context until the capability is
    # invoked. Once invoked, keep it through the provider request that consumes
    # the real result so behavioral contracts remain visible during synthesis.
    if not underlying_ids:
        return True
    for msg in messages[describe_index + 1:]:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and str(msg.get("tool_call_id") or "") in underlying_ids
        ):
            return not _receipt_consumed(msg)
    return True


def _protected_assistant_payload(msg: Dict[str, Any]) -> bool:
    protected = (
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "thinking",
        "anthropic_content_blocks",
        "codex_reasoning_items",
        "codex_message_items",
        "provider_data",
        "api_content",
    )
    return any(msg.get(key) not in (None, "", [], {}) for key in protected)


def edit_consumed_transients(
    messages: Iterable[Dict[str, Any]],
    *,
    keep_recent: int = 3,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Bound consumed capability disclosures while keeping recent context.

    ``mark_tool_results_consumed`` stamps a result only after the provider
    accepted the request containing it. Keep the latest configured number of
    consumed discovery results so the model can reuse a just-found schema
    instead of searching again on the next provider call. Older pairs can then
    be removed from the provider projection. Provider-signed assistant
    envelopes keep a minimum result placeholder instead of having their call
    removed underneath them.
    """
    source = [copy.deepcopy(msg) for msg in messages]
    transient_ids = transient_call_ids(source)
    consumed_order = [
        str(msg.get("tool_call_id") or "")
        for msg in source
        if (
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and str(msg.get("tool_call_id") or "") in transient_ids
            and _receipt_consumed(msg)
        )
    ]
    retained_ids = set(
        consumed_order[-max(0, int(keep_recent)):]
        if keep_recent
        else ()
    )
    consumed_ids = {
        call_id
        for call_id in consumed_order
        if call_id not in retained_ids
        if not _described_contract_pending(
            source,
            describe_call_id=call_id,
        )
    }
    if not consumed_ids:
        return source, []

    protected_ids: Set[str] = set()
    for msg in source:
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if (
            isinstance(calls, list)
            and msg.get("role") == "assistant"
            and _protected_assistant_payload(msg)
        ):
            protected_ids.update(
                call_id
                for call in calls
                for call_id, _name, _arguments in [_call_parts(call)]
                if call_id in consumed_ids
            )

    report: List[Dict[str, str]] = []
    edited: List[Dict[str, Any]] = []
    for msg in source:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            if call_id in consumed_ids:
                if call_id in protected_ids:
                    msg["content"] = (
                        "[capability disclosure consumed; payload removed]"
                    )
                    report.append(
                        {"tool_call_id": call_id, "action": "placeholder"}
                    )
                    edited.append(msg)
                else:
                    report.append(
                        {"tool_call_id": call_id, "action": "remove_result"}
                    )
                continue

        calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and isinstance(calls, list):
            kept = []
            removed = []
            for call in calls:
                call_id, _name, _arguments = _call_parts(call)
                if call_id in consumed_ids and call_id not in protected_ids:
                    removed.append(call_id)
                else:
                    kept.append(call)
            if removed:
                if kept:
                    msg["tool_calls"] = kept
                elif msg.get("content") not in (None, ""):
                    msg.pop("tool_calls", None)
                else:
                    report.extend(
                        {"tool_call_id": call_id, "action": "remove_call"}
                        for call_id in removed
                    )
                    continue
                report.extend(
                    {"tool_call_id": call_id, "action": "remove_call"}
                    for call_id in removed
                )
        edited.append(msg)
    return edited, report


def _resolved_durable_call(call: Any) -> Any:
    """Represent a real generic ``tool_call`` as the resolved tool in history."""
    call_id, name, raw_arguments = _call_parts(call)
    if name != "tool_call":
        return call
    wrapper = _arguments_object(raw_arguments)
    resolved_name = str(wrapper.get("name") or "").strip()
    resolved_args = wrapper.get("arguments", {})
    if not resolved_name or resolved_name in _TRANSIENT_UNDERLYING:
        return call
    if isinstance(resolved_args, str):
        encoded_args = resolved_args
    else:
        encoded_args = json.dumps(resolved_args, ensure_ascii=False, separators=(",", ":"))
    try:
        from tools.registry import registry

        schema = registry.get_schema(resolved_name) or {}
        schema_hash = hashlib.sha256(
            json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    except Exception:
        schema_hash = ""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": resolved_name, "arguments": encoded_args},
        "resolved_via": "tool_call",
        "schema_hash": schema_hash,
    }


def project_message(
    msg: Dict[str, Any],
    *,
    transient_ids: Set[str],
) -> Optional[Dict[str, Any]]:
    """Return one durable message clone, or ``None`` when fully transient."""
    if not isinstance(msg, dict):
        return None
    if msg.get("_capability_transient"):
        return None
    if msg.get("role") == "tool" and str(msg.get("tool_call_id") or "") in transient_ids:
        return None

    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return msg

    kept: List[Any] = []
    for call in calls:
        call_id, name, arguments = _call_parts(call)
        if call_id in transient_ids or _is_transient_call(name, arguments):
            continue
        kept.append(_resolved_durable_call(call))

    if len(kept) == len(calls):
        # Generic real calls still need their durable resolved identity.
        resolved = [_resolved_durable_call(call) for call in calls]
        if all(a is b for a, b in zip(resolved, calls)):
            return msg
        clone = copy.copy(msg)
        clone["tool_calls"] = resolved
        return clone

    content = msg.get("content")
    if not kept and (content is None or content == ""):
        return None
    clone = copy.copy(msg)
    clone["tool_calls"] = kept
    clone["_capability_projection"] = True
    return clone


def durable_projection(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source = list(messages)
    ids = transient_call_ids(source)
    result: List[Dict[str, Any]] = []
    for msg in source:
        projected = project_message(msg, transient_ids=ids)
        if projected is not None:
            result.append(projected)
    return result
