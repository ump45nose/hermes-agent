import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from tools.delegate_tool import _run_single_child, _write_research_evidence_bundle


def _child(tmp_path):
    return SimpleNamespace(_research_artifact_dir=tmp_path)


def test_markdown_research_handoff_does_not_replay_full_report(tmp_path):
    unique_body = "FULL-REPORT-ONLY-" + ("x" * 20_000)
    summary = (
        "# Findings\n"
        "- Claim one https://example.test/one\n"
        "- Claim two https://example.test/two\n"
        + unique_body
    )
    envelope, evidence_path, evidence_sha, handoff_path, handoff_sha = (
        _write_research_evidence_bundle(
            _child(tmp_path),
            goal="research",
            summary=summary,
            messages=[{"role": "assistant", "content": summary}],
            status="completed",
        )
    )
    parsed = json.loads(envelope)
    assert len(envelope) <= 12_000
    assert unique_body not in envelope
    assert parsed["kind"] == "research_leaf_handoff"
    assert parsed["handoff_mode"] == "deterministic_markdown_extract"
    assert parsed["artifact_is_audit_only"] is True
    assert json.loads(open(evidence_path, encoding="utf-8").read())["report"] == summary
    assert open(handoff_path, encoding="utf-8").read() == envelope
    assert hashlib.sha256(open(evidence_path, "rb").read()).hexdigest() == evidence_sha
    assert hashlib.sha256(open(handoff_path, "rb").read()).hexdigest() == handoff_sha


def test_structured_research_handoff_preserves_bounded_fields(tmp_path):
    summary = json.dumps(
        {
            "claims": ["claim-a", "claim-b"],
            "source_ids": ["s1", "s2"],
            "contradictions": ["conflict"],
            "unexpected_findings": ["surprise"],
            "unresolved": ["gap"],
        }
    )
    envelope, _, _, _, _ = _write_research_evidence_bundle(
        _child(tmp_path),
        goal="research",
        summary=summary,
        messages=[],
        status="completed",
    )
    parsed = json.loads(envelope)
    assert parsed["handoff_mode"] == "structured_json"
    assert parsed["claims"] == ["claim-a", "claim-b"]
    assert parsed["source_ids"] == ["s1", "s2"]
    assert parsed["contradictions"] == ["conflict"]
    assert parsed["unexpected_findings"] == ["surprise"]
    assert parsed["unresolved"] == ["gap"]


def test_failed_research_handoff_keeps_failure_unresolved(tmp_path):
    envelope, _, _, _, _ = _write_research_evidence_bundle(
        _child(tmp_path),
        goal="research",
        summary="leaf timed out",
        messages=[],
        status="failed",
    )
    parsed = json.loads(envelope)
    assert parsed["status"] == "failed"
    assert "failed" in parsed["unresolved"]


def test_research_leaf_max_iterations_is_partial_with_trace_summary_only(tmp_path):
    child = MagicMock()
    child.runtime_role = "research_leaf"
    child._delegate_role = "leaf"
    child._credential_pool = None
    child._research_artifact_dir = tmp_path
    child._delegate_saved_tool_names = []
    child._subagent_id = "research-leaf-test"
    child.tool_progress_callback = None
    child.model = "research-model"
    child.session_prompt_tokens = 1200
    child.session_completion_tokens = 300
    child.session_estimated_cost_usd = 0.01
    child.session_reasoning_tokens = 0
    child.run_conversation.return_value = {
        "final_response": json.dumps(
            {
                "claims": ["bounded claim"],
                "source_ids": ["source-1"],
                "contradictions": [],
                "unexpected_findings": [],
                "unresolved": ["needs another source"],
            }
        ),
        "completed": False,
        "interrupted": False,
        "api_calls": 5,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "one",
                        "function": {
                            "name": "mcp__smart_search__smart_fetch",
                            "arguments": '{"url":"https://example.test/one"}',
                        },
                    },
                    {
                        "id": "two",
                        "function": {
                            "name": "mcp__smart_search__smart_search",
                            "arguments": '{"query":"two"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "one",
                "content": '{"ok":true}',
            },
            {
                "role": "tool",
                "tool_call_id": "two",
                "content": '{"ok":false,"error":"rate limited"}',
            },
        ],
    }
    parent = MagicMock()
    parent._current_task_id = None

    entry = _run_single_child(
        task_index=0,
        goal="research",
        child=child,
        parent_agent=parent,
    )

    assert entry["status"] == "partial"
    assert entry["exit_reason"] == "max_iterations"
    assert "tool_trace" not in entry
    trace = entry["tool_trace_summary"]
    assert trace["calls"] == 2
    assert trace["errors"] == 1
    assert trace["unique_tools"] == [
        "mcp__smart_search__smart_fetch",
        "mcp__smart_search__smart_search",
    ]
    assert trace["bytes"] > 0
    assert set(trace) == {"calls", "errors", "unique_tools", "bytes"}
    assert entry["evidence_bundle_path"] == str(tmp_path / "evidence-bundle.json")
    handoff = json.loads(entry["summary"])
    assert handoff["status"] == "partial"
    assert "partial" in handoff["unresolved"]
    evidence = json.loads(
        (tmp_path / "evidence-bundle.json").read_text(encoding="utf-8")
    )
    assert len(evidence["messages"]) == 3
