"""Provider-safe pruning of already-consumed tool result bodies."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from agent.tool_result_classification import tool_may_have_side_effect


@dataclass
class ToolResultReceipt:
    tool_name: str
    result_status: str = "unknown"
    effect: str = "unknown"
    artifact_ref: str | None = None
    consumed_turn: int | None = None
    steer_present: bool = False
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content or "")


def artifact_ref_from_content(content: Any) -> str | None:
    text = _text(content)
    marker = "Full output saved to:"
    for line in text.splitlines():
        if line.startswith(marker):
            return line[len(marker):].strip() or None
    return None


def build_receipt(
    tool_name: str,
    content: Any,
    *,
    result_status: str = "unknown",
    effect: str | None = None,
    artifact_ref: str | None = None,
    supersedes: str | None = None,
) -> ToolResultReceipt:
    text = _text(content)
    status = result_status
    if not text.strip() and status == "unknown":
        status = "empty"
    resolved_effect = effect or (
        "unknown" if tool_may_have_side_effect(tool_name) else "none"
    )
    return ToolResultReceipt(
        tool_name=tool_name,
        result_status=status,
        effect=resolved_effect,
        artifact_ref=artifact_ref,
        steer_present="/steer" in text or "<steer" in text,
        supersedes=supersedes,
    )


def receipt_from_message(message: dict[str, Any]) -> ToolResultReceipt:
    raw = message.get("_tool_receipt")
    if isinstance(raw, dict):
        allowed = ToolResultReceipt.__dataclass_fields__
        receipt = ToolResultReceipt(
            **{key: raw[key] for key in raw if key in allowed}
        )
        if "/steer" in _text(message.get("content")) or "<steer" in _text(
            message.get("content")
        ):
            receipt.steer_present = True
        return receipt
    disposition = message.get("effect_disposition")
    effect = disposition if disposition in {"none", "landed", "unknown"} else None
    return build_receipt(
        str(message.get("tool_name") or message.get("name") or ""),
        message.get("content"),
        effect=effect,
    )


def _call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or call.get("call_id") or "")
    return str(getattr(call, "id", "") or getattr(call, "call_id", "") or "")


def _current_unconsumed_ids(messages: list[dict[str, Any]]) -> set[str]:
    """IDs in the latest assistant tool-call batch at the request tail."""
    for message in reversed(messages):
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            return {_call_id(call) for call in message["tool_calls"]}
        if role == "assistant" and _text(message.get("content")).strip():
            break
    return set()


def _safe_to_clear(
    receipt: ToolResultReceipt,
    *,
    duplicate: bool,
    retry_succeeded: bool,
    phase: str,
) -> bool:
    if receipt.steer_present:
        return False
    if receipt.effect != "none":
        return False
    if receipt.result_status in {"empty", "cancelled"}:
        return True
    if receipt.result_status == "error":
        return retry_succeeded and phase in {"failures", "active"}
    return duplicate or bool(receipt.supersedes)


def edit_tool_context(
    messages: list[dict[str, Any]],
    *,
    report_only: bool = False,
    phase: str = "active",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return edited API messages and a structured audit report.

    The newest tool batch is always preserved for one model consumption.
    Older safe results are removed together with a single-call assistant pair;
    mixed-call assistant turns keep the call and receive a minimum placeholder.
    """
    edited = copy.deepcopy(messages)
    current_ids = _current_unconsumed_ids(edited)
    seen_success_by_name: set[str] = set()
    seen_payloads: set[tuple[str, str]] = set()
    report: list[dict[str, Any]] = []

    # Map result id -> assistant index and call count.
    owners: dict[str, tuple[int, int]] = {}
    for index, message in enumerate(edited):
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(calls, list):
            for call in calls:
                owners[_call_id(call)] = (index, len(calls))

    remove_indices: set[int] = set()
    for index in range(len(edited) - 1, -1, -1):
        message = edited[index]
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        receipt = receipt_from_message(message)
        name = receipt.tool_name
        payload_hash = hashlib.sha256(_text(message.get("content")).encode()).hexdigest()
        key = (name, payload_hash)
        if call_id in current_ids:
            if receipt.result_status == "success":
                seen_success_by_name.add(name)
            seen_payloads.add(key)
            continue
        duplicate = key in seen_payloads
        retry_succeeded = name in seen_success_by_name
        if receipt.result_status == "success":
            seen_success_by_name.add(name)
        seen_payloads.add(key)
        if not _safe_to_clear(
            receipt,
            duplicate=duplicate,
            retry_succeeded=retry_succeeded,
            phase=phase,
        ):
            if (
                phase == "active"
                and not receipt.steer_present
                and receipt.effect in {"landed", "unknown"}
            ):
                message["content"] = (
                    "[tool action receipt after consumption: "
                    f"status={receipt.result_status}; effect={receipt.effect}; "
                    f"artifact={receipt.artifact_ref or 'none'}]"
                )
                report.append(
                    {
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "action": "action_receipt",
                        "status": receipt.result_status,
                    }
                )
            continue
        owner = owners.get(call_id)
        action = "placeholder"
        if owner and owner[1] == 1:
            action = "remove_pair"
            remove_indices.update({index, owner[0]})
        else:
            message["content"] = (
                "[tool result body removed after consumption; "
                f"status={receipt.result_status}; effect={receipt.effect}]"
            )
        report.append(
            {
                "tool_call_id": call_id,
                "tool_name": name,
                "action": action,
                "status": receipt.result_status,
            }
        )
    if report_only:
        return messages, report
    return [
        message for index, message in enumerate(edited) if index not in remove_indices
    ], report


def strip_internal_tool_metadata(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        message.pop("_tool_receipt", None)
        message.pop("effect_disposition", None)


def mark_tool_results_consumed(
    messages: list[dict[str, Any]],
    *,
    consumed_turn: int,
    session_db: Any = None,
    session_id: str = "",
) -> None:
    """Mark the latest result batch after a provider accepted its request."""
    current_ids = _current_unconsumed_ids(messages)
    for message in messages:
        if (
            message.get("role") != "tool"
            or str(message.get("tool_call_id") or "") not in current_ids
        ):
            continue
        receipt = receipt_from_message(message)
        receipt.consumed_turn = consumed_turn
        receipt.steer_present = receipt.steer_present or (
            "/steer" in _text(message.get("content"))
        )
        message["_tool_receipt"] = receipt.to_dict()
        updater = getattr(session_db, "update_tool_receipt", None)
        if callable(updater) and session_id:
            try:
                updater(session_id, str(message.get("tool_call_id") or ""), receipt.to_dict())
            except Exception:
                pass
