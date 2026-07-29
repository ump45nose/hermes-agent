#!/usr/bin/env python3
"""Inspect real APISIX LLM wire requests without reading request headers."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LOG = Path("/vol2/1000/Docker/APISIX/logs/llm-requests.jsonl")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:sk|ghp|github_pat|m0)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r'(?i)("(?:authorization|api[_-]?key|token)"\s*:\s*")([^"]+)(")'
    ),
)


def redact(value: str) -> str:
    rendered = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 3:
            rendered = pattern.sub(r"\1[REDACTED]\3", rendered)
        else:
            rendered = pattern.sub("[REDACTED]", rendered)
    return rendered


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def summarize(line_number: int, record: dict[str, Any]) -> dict[str, Any]:
    try:
        body = json.loads(record.get("request_body") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        body = {}
    messages = body.get("messages") or body.get("input") or []
    if not isinstance(messages, list):
        messages = []
    system_chars = 0
    tool_chars = 0
    tools: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "chars": 0,
            "effects": defaultdict(int),
            "forms": defaultdict(int),
        }
    )
    agents_markers = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = _content_text(message.get("content") or "")
        if role in {"system", "developer"}:
            system_chars += len(content)
        agents_markers += content.count("AGENTS.md")
        if role != "tool":
            continue
        name = str(message.get("name") or message.get("tool_name") or "<unknown>")
        effect = str(message.get("effect_disposition") or "<wire-stripped>")
        tools[name]["count"] += 1
        tools[name]["chars"] += len(content)
        tools[name]["effects"][effect] += 1
        if "tool read result consumed; body cleared" in content:
            form = "read_receipt"
        elif "tool action receipt after consumption" in content:
            form = "action_receipt"
        elif "unresolved tool blocker after consumption" in content:
            form = "blocker_receipt"
        elif "tool result body removed after consumption" in content:
            form = "removed_placeholder"
        else:
            form = "full_body"
        tools[name]["forms"][form] += 1
        tool_chars += len(content)
    return {
        "line": line_number,
        "timestamp": record.get("timestamp"),
        "request_id": record.get("request_id"),
        "uri": record.get("uri"),
        "model": body.get("model"),
        "status": record.get("status"),
        "request_bytes": record.get("request_bytes"),
        "response_bytes": record.get("response_bytes"),
        "system_chars": system_chars,
        "message_count": len(messages),
        "tool_result_count": sum(item["count"] for item in tools.values()),
        "tool_result_chars": tool_chars,
        "agents_md_markers": agents_markers,
        "tools": {
            name: {
                "count": item["count"],
                "chars": item["chars"],
                "effects": dict(item["effects"]),
                "forms": dict(item["forms"]),
            }
            for name, item in sorted(
                tools.items(),
                key=lambda pair: (-pair[1]["chars"], pair[0]),
            )
        },
    }


def _records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                records.append((line_number, record))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect owner-only APISIX LLM request-body logs."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--contains", default="", help="Request-body substring filter")
    parser.add_argument("--model", default="", help="Exact model filter")
    parser.add_argument("--request-id", default="", help="Exact APISIX request id")
    parser.add_argument("--last", type=int, default=1, help="Number of matches")
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Include auxiliary conversation-title requests",
    )
    parser.add_argument(
        "--show-system",
        action="store_true",
        help="Print exact system/developer message content after the summary",
    )
    parser.add_argument(
        "--show-request",
        action="store_true",
        help="Print the complete redacted request JSON after the summary",
    )
    args = parser.parse_args()

    matches: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for line_number, record in _records(args.log):
        raw = str(record.get("request_body") or "")
        if args.contains and args.contains not in raw:
            continue
        if args.request_id and record.get("request_id") != args.request_id:
            continue
        try:
            body = json.loads(raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        if args.model and body.get("model") != args.model:
            continue
        messages = body.get("messages") or []
        is_title = bool(
            messages
            and isinstance(messages[0], dict)
            and "Generate a short, descriptive title"
            in _content_text(messages[0].get("content") or "")
        )
        if is_title and not args.include_title:
            continue
        matches.append((line_number, record, body))

    selected = matches[-max(1, args.last):]
    print(
        json.dumps(
            [summarize(line, record) for line, record, _body in selected],
            ensure_ascii=False,
            indent=2,
        )
    )
    for line_number, record, body in selected:
        if args.show_system:
            print(f"\n===== line {line_number} system/developer =====")
            for message in body.get("messages") or []:
                if (
                    isinstance(message, dict)
                    and message.get("role") in {"system", "developer"}
                ):
                    print(redact(_content_text(message.get("content") or "")))
        if args.show_request:
            print(f"\n===== line {line_number} redacted request =====")
            print(redact(json.dumps(body, ensure_ascii=False, indent=2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
