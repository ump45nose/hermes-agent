import hashlib
import json
from types import SimpleNamespace

from tools.delegate_tool import _write_research_evidence_bundle


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
