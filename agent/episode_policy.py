"""Knowledge-layer policy for local Episode extraction and promotion."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


EXCLUDED_SOURCES = frozenset({"cron", "subagent", "research_leaf", "test", "pytest"})
_TEST_MARKERS = re.compile(r"(?:^|[/_:.-])(test|pytest|fixture)(?:$|[/_:.-])", re.I)
_DYNAMIC_INFRA = re.compile(
    r"(?:\b(?:\d{1,3}\.){3}\d{1,3}\b|"
    r"\b(?:port|端口)\s*[:=]?\s*\d{2,5}\b|"
    r"\b(?:docker|container|tailscale|dns|systemd)\b.{0,30}"
    r"\b(?:running|stopped|up|down|enabled|disabled)\b|"
    r"(?:容器|服务|网关|代理).{0,20}(?:运行中|已停止|端口|地址|状态))",
    re.I,
)


def eligible_session(
    *,
    source: str,
    session_id: str,
    ended_at: Any,
    message_count: int,
) -> bool:
    normalized = str(source or "").strip().lower()
    if normalized in EXCLUDED_SOURCES or _TEST_MARKERS.search(normalized):
        return False
    if _TEST_MARKERS.search(str(session_id or "")):
        return False
    return ended_at is not None and int(message_count or 0) >= 6


def episode_input_messages(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Return user/assistant text plus deterministic tool receipts only."""
    rendered: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "get"):
            getter = row.get
        else:
            def getter(key: str, default: Any = None, _row: Any = row) -> Any:
                try:
                    return _row[key]
                except (KeyError, IndexError):
                    return default
        role = str(getter("role", "") or "")
        if role == "system":
            continue
        if role == "tool":
            raw = getter("tool_receipt")
            try:
                receipt = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                receipt = None
            if not isinstance(receipt, dict):
                continue
            rendered.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            key: receipt.get(key)
                            for key in (
                                "tool_name",
                                "result_status",
                                "effect",
                                "artifact_ref",
                            )
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
            continue
        if role not in {"user", "assistant"}:
            continue
        content = getter("content", "")
        try:
            from agent.controller_protocol import is_controller_receipt_text

            if is_controller_receipt_text(content):
                continue
        except Exception:
            pass
        if isinstance(content, str) and content.strip():
            rendered.append({"role": role, "content": content})
    return rendered


def contains_dynamic_infrastructure(value: str) -> bool:
    return bool(_DYNAMIC_INFRA.search(str(value or "")))
