"""One-shot capture of the exact provider-bound request body."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home


_SECRET_KEY = re.compile(
    r"(authorization|api[_-]?key|token|password|secret|cookie)", re.I
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return str(value)


def _body_only(kwargs: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: _jsonable(value)
        for key, value in kwargs.items()
        if key not in {"extra_headers", "headers", "http_headers"}
    }
    return body


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if _SECRET_KEY.search(str(key)) and isinstance(item, str)
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _BEARER.sub("Bearer [REDACTED]", value)
    return value


def _url_path(agent: Any) -> str:
    mode = getattr(agent, "api_mode", "")
    if mode == "codex_responses":
        return "/v1/responses"
    if mode == "anthropic_messages":
        return "/v1/messages"
    if mode == "bedrock_converse":
        return "/model/{modelId}/converse-stream"
    return "/v1/chat/completions"


def capture_request_snapshot(
    agent: Any,
    api_kwargs: dict[str, Any],
    *,
    request_id: str,
) -> Path | None:
    config_path = get_hermes_home() / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return None
    observability = config.get("observability") or {}
    if not isinstance(observability, dict):
        return None
    mode = observability.get("request_snapshot", "off")
    if mode not in {"once", "on"}:
        return None
    body = _body_only(api_kwargs)
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(raw.encode()).hexdigest()
    snapshot_id = request_id or uuid.uuid4().hex
    directory = get_hermes_home() / "observability" / "request-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    raw_path = directory / f"{snapshot_id}.wire.json"
    display_path = directory / f"{snapshot_id}.redacted.json"
    manifest_path = directory / f"{snapshot_id}.manifest.json"
    envelope = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "request_id": snapshot_id,
        "url_path": _url_path(agent),
        "base_url": str(getattr(agent, "base_url", "") or ""),
        "body": body,
    }
    raw_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    display = {**envelope, "body": _redact(body)}
    display_path.write_text(
        json.dumps(display, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "request_id": snapshot_id,
        "sha256": sha,
        "wire_file": raw_path.name,
        "redacted_file": display_path.name,
        "sources": list(getattr(agent, "_prompt_source_manifest", []) or [])
        + ([{"kind": "scenario"}] if getattr(agent, "_last_scenario_context", "") else [])
        + [
            {"kind": "history"},
            {"kind": "tool_result"},
            {"kind": "tool_schema", "count": len(getattr(agent, "tools", []) or [])},
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for path in (raw_path, display_path, manifest_path):
        os.chmod(path, 0o600)
    if mode == "once":
        observability["request_snapshot"] = "off"
        config["observability"] = observability
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.chmod(config_path, 0o600)
    return raw_path
