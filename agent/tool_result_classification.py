"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


# Tools whose interrupted/dangling execution is safe to discard because they
# cannot mutate either external state or Hermes session state. Unknown/plugin/
# MCP tools stay effect-capable by default.
NO_EFFECT_TOOL_NAMES = frozenset({
    "read_file", "read_tool_artifact", "search_files", "session_search",
    "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})

# These MCP servers expose retrieval-only capabilities in Hermes.  Treating
# every dynamically-named MCP tool as effect-capable is the safe general
# default, but it also caused successful SmartSearch results to carry
# ``effect=unknown`` forever.  That prevented the Tool Context Editor from
# replacing already-consumed 20-80K search/fetch bodies with receipts.
#
# Prefixes belong here only when the whole exposed server is read-only.  A
# mixed read/write MCP server must keep the default ``unknown`` disposition
# and classify individual tools through an explicit allowlist instead.
NO_EFFECT_TOOL_PREFIXES = (
    "mcp__smart_search__",
    "mcp__context7__",
)


def declared_tool_effect(tool_name: str) -> str | None:
    """Return authoritative registry/MCP effect metadata when available."""
    try:
        from tools.registry import registry

        entry = registry.get_entry(tool_name)
    except Exception:
        entry = None
    effect = getattr(entry, "effect_disposition", None)
    return effect if effect in {"none", "unknown"} else None


def declared_tool_retention(tool_name: str) -> str | None:
    """Return a tool name that must be called before this body can be pruned."""
    try:
        from tools.registry import registry

        entry = registry.get_entry(tool_name)
    except Exception:
        entry = None
    value = getattr(entry, "retain_result_until", None)
    return str(value) if value else None


def tool_may_have_side_effect(tool_name: str) -> bool:
    declared = declared_tool_effect(tool_name)
    if declared is not None:
        return declared != "none"
    return (
        tool_name not in NO_EFFECT_TOOL_NAMES
        and not any(tool_name.startswith(prefix) for prefix in NO_EFFECT_TOOL_PREFIXES)
    )


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False
