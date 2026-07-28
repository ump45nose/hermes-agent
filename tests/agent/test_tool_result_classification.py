"""Tests for shared tool result classification helpers."""

import json

from agent.tool_result_classification import (
    declared_tool_retention,
    file_mutation_result_landed,
    tool_may_have_side_effect,
)


def test_write_file_with_nested_lint_error_counts_as_landed():
    result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })

    assert file_mutation_result_landed("write_file", result) is True


def test_patch_with_nested_lsp_diagnostics_counts_as_landed():
    result = json.dumps({
        "success": True,
        "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
        "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
    })

    assert file_mutation_result_landed("patch", result) is True


def test_top_level_file_mutation_error_does_not_count_as_landed():
    result = json.dumps({"success": True, "error": "post-write verification failed"})

    assert file_mutation_result_landed("patch", result) is False


def test_side_effect_classification_keeps_session_mutations():
    assert tool_may_have_side_effect("todo") is True
    assert tool_may_have_side_effect("memory") is True
    assert tool_may_have_side_effect("write_file") is True
    assert tool_may_have_side_effect("mcp_unknown") is True
    assert tool_may_have_side_effect("read_file") is False
    assert tool_may_have_side_effect("web_search") is False


def test_registry_effect_metadata_overrides_legacy_name_classification():
    from tools.registry import registry

    name = "test_declared_read_only_effect"
    registry.register(
        name=name,
        toolset="test",
        schema={
            "name": name,
            "description": "read-only test tool",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda _args, **_kwargs: "{}",
        effect_disposition="none",
        retain_result_until="next_tool",
    )
    try:
        assert tool_may_have_side_effect(name) is False
        assert declared_tool_retention(name) == "next_tool"
    finally:
        registry.deregister(name)
