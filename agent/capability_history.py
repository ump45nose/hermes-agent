"""Durable projection for progressive capability-disclosure messages.

The live API tool loop keeps search/describe/Skill-load calls so the model can
use their results within the current user request. Durable history must not:
those payloads are discovery scaffolding, not conversation state.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


_TRANSIENT_BRIDGES = {"tool_search", "tool_describe", "skill_search"}
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
