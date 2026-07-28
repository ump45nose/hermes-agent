"""Read a bounded slice from the current session's tool-result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry


_MAX_READ_CHARS = 12_000
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


READ_TOOL_ARTIFACT_SCHEMA = {
    "name": "read_tool_artifact",
    "description": (
        "Read a bounded character slice from an owner-only tool-result artifact "
        "created in this session. Use the exact artifact path from a consumed "
        "tool receipt. This tool cannot read arbitrary files or artifacts from "
        "another session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artifact_ref": {
                "type": "string",
                "description": "Exact artifact path shown in a tool receipt.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Zero-based character offset.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_READ_CHARS,
                "default": 8_000,
                "description": "Maximum characters to return.",
            },
        },
        "required": ["artifact_ref"],
        "additionalProperties": False,
    },
}


def read_tool_artifact(args: dict[str, Any], *, session_id: str = "", **_: Any) -> str:
    """Return one bounded slice, restricted to the caller's session directory."""
    if not session_id:
        return json.dumps(
            {"ok": False, "error_type": "missing_session", "error": "session id unavailable"},
            ensure_ascii=False,
        )

    artifact_ref = str(args.get("artifact_ref") or "").strip()
    if not artifact_ref:
        return json.dumps(
            {"ok": False, "error_type": "invalid_argument", "error": "artifact_ref is required"},
            ensure_ascii=False,
        )

    try:
        offset = max(0, int(args.get("offset", 0)))
        limit = min(_MAX_READ_CHARS, max(1, int(args.get("limit", 8_000))))
    except (TypeError, ValueError):
        return json.dumps(
            {"ok": False, "error_type": "invalid_argument", "error": "offset and limit must be integers"},
            ensure_ascii=False,
        )

    session_root = (
        get_hermes_home()
        / "artifacts"
        / "tool-results"
        / str(session_id)
    ).resolve()
    try:
        candidate = Path(artifact_ref).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return json.dumps(
            {"ok": False, "error_type": "not_found", "error": "artifact not found"},
            ensure_ascii=False,
        )

    if not candidate.is_relative_to(session_root) or not candidate.is_file():
        return json.dumps(
            {
                "ok": False,
                "error_type": "scope_violation",
                "error": "artifact is outside the current session",
            },
            ensure_ascii=False,
        )
    try:
        artifact_stat = candidate.stat()
    except OSError:
        return json.dumps(
            {"ok": False, "error_type": "not_found", "error": "artifact not found"},
            ensure_ascii=False,
        )
    if artifact_stat.st_mode & 0o077:
        return json.dumps(
            {
                "ok": False,
                "error_type": "unsafe_permissions",
                "error": "artifact is not owner-only",
            },
            ensure_ascii=False,
        )
    size_bytes = artifact_stat.st_size
    if size_bytes > _MAX_ARTIFACT_BYTES:
        return json.dumps(
            {
                "ok": False,
                "error_type": "artifact_too_large",
                "error": "artifact exceeds the safe read limit",
                "size_bytes": size_bytes,
            },
            ensure_ascii=False,
        )

    try:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return json.dumps(
            {"ok": False, "error_type": "read_failed", "error": "artifact could not be read"},
            ensure_ascii=False,
        )

    total_chars = len(content)
    page = content[offset: offset + limit]
    next_offset = offset + len(page)
    return json.dumps(
        {
            "ok": True,
            "artifact_ref": str(candidate),
            "offset": offset,
            "returned_chars": len(page),
            "total_chars": total_chars,
            "next_offset": next_offset if next_offset < total_chars else None,
            "content": page,
        },
        ensure_ascii=False,
    )


registry.register(
    name="read_tool_artifact",
    toolset="tool_artifact",
    schema=READ_TOOL_ARTIFACT_SCHEMA,
    handler=read_tool_artifact,
    emoji="🧾",
    max_result_size_chars=_MAX_READ_CHARS + 2_000,
)
