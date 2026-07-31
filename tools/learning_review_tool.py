"""Owner-gated review of Episode-derived Skill/Tool candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from agent.episode_memory import subject_id
from hermes_state import SessionDB
from tools.registry import registry, tool_error


def _session_value(name: str) -> str:
    try:
        from gateway.session_context import get_session_env

        return str(get_session_env(name, "") or "")
    except Exception:
        return ""


def _active_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _root_for(home: Path) -> Path:
    if home.parent.name == "profiles":
        return home.parent.parent
    return home


def _review_config(home: Path) -> Dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load((home / "config.yaml").read_text()) or {}
        return ((value.get("episode_memory") or {}).get("distillation") or {})
    except Exception:
        return {}


def _authorized(home: Path, config: Dict[str, Any]) -> bool:
    if home.name != "lingjun":
        return False
    platform = _session_value("HERMES_SESSION_PLATFORM") or _session_value(
        "HERMES_SESSION_SOURCE"
    )
    user_id = _session_value("HERMES_SESSION_USER_ID")
    if not platform or not user_id:
        return False
    reviewer = subject_id(platform, user_id)
    return reviewer in {
        str(value) for value in (config.get("reviewer_subjects") or [])
    }


def _reviewer_subject() -> str:
    platform = _session_value("HERMES_SESSION_PLATFORM") or _session_value(
        "HERMES_SESSION_SOURCE"
    )
    user_id = _session_value("HERMES_SESSION_USER_ID")
    if not platform or not user_id:
        return ""
    return subject_id(platform, user_id)


def _available() -> bool:
    home = _active_home()
    config = _review_config(home)
    return bool(
        home.name == "lingjun"
        and config.get("enabled") is True
        and config.get("reviewer_subjects")
    )


def _candidate_view(profile: str, row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = json.loads(row.get("payload_json") or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "profile": profile,
        "candidate_id": int(row["id"]),
        "kind": row["kind"],
        "semantic_key": row["semantic_key"],
        "title": row["title"],
        "status": row["status"],
        "version": row["version_hash"],
        "successful_run_count": int(row.get("successful_run_count") or 0),
        "repeated_failure_count": int(row.get("repeated_failure_count") or 0),
        "payload": payload,
    }


def learning_review(args: Dict[str, Any], **_: Any) -> str:
    home = _active_home()
    config = _review_config(home)
    if not _authorized(home, config):
        return tool_error(
            "learning_review 只允许配置的 Lingjun owner 在前台会话中使用。",
            success=False,
        )
    action = str(args.get("action") or "").lower()
    allowed_profiles = {
        str(value)
        for value in (config.get("reviewer_profiles") or ["lingjun"])
    }
    requested_profile = str(args.get("profile") or "").strip()
    root = _root_for(home)

    if action == "list":
        profiles = [requested_profile] if requested_profile else sorted(allowed_profiles)
        result = []
        for profile in profiles:
            if profile not in allowed_profiles:
                continue
            path = root / "profiles" / profile / "state.db"
            if not path.exists():
                continue
            db = SessionDB(db_path=path)
            try:
                rows = db.list_learning_candidates(
                    statuses=(
                        "ready_for_review",
                        "approved",
                        "applying",
                        "failed",
                    ),
                    limit=50,
                )
                result.extend(_candidate_view(profile, dict(row)) for row in rows)
            finally:
                db.close()
        return json.dumps({"success": True, "candidates": result}, ensure_ascii=False)

    if action not in {
        "get",
        "approve",
        "reject",
        "claim",
        "complete",
        "fail",
    }:
        return tool_error(
            "action 必须是 list/get/approve/reject/claim/complete/fail。",
            success=False,
        )
    if requested_profile not in allowed_profiles:
        return tool_error("目标 Profile 不在 reviewer_profiles 权限内。", success=False)
    candidate_id = int(args.get("candidate_id") or 0)
    version = str(args.get("version") or "")
    path = root / "profiles" / requested_profile / "state.db"
    if not path.exists() or candidate_id <= 0:
        return tool_error("候选不存在。", success=False)
    db = SessionDB(db_path=path)
    try:
        rows = db.list_learning_candidates(
            statuses=(
                "observe",
                "ready_for_review",
                "approved",
                "applying",
                "applied",
                "failed",
                "rejected",
            ),
            limit=500,
        )
        row = next((dict(value) for value in rows if int(value["id"]) == candidate_id), None)
        if row is None:
            return tool_error("候选不存在或已 superseded。", success=False)
        if action == "get":
            return json.dumps(
                {"success": True, "candidate": _candidate_view(requested_profile, row)},
                ensure_ascii=False,
            )
        if not version:
            return tool_error("审核和应用结果必须携带候选 version。", success=False)
        if action == "claim":
            claimed = db.claim_learning_candidate_application(
                candidate_id,
                version_hash=version,
                reviewer=_reviewer_subject(),
            )
            if claimed is None:
                return tool_error(
                    "候选未获批准、版本已变化或仍被有效租约占用。",
                    success=False,
                )
            candidate = _candidate_view(requested_profile, claimed)
            candidate["claim_id"] = str(
                claimed.get("application_claim_id") or ""
            )
            candidate["lease_until"] = claimed.get(
                "application_lease_until"
            )
            return json.dumps(
                {"success": True, "candidate": candidate},
                ensure_ascii=False,
            )
        if action in {"complete", "fail"}:
            result = args.get("result")
            claim_id = str(args.get("claim_id") or "").strip()
            if not isinstance(result, dict) or not claim_id:
                return tool_error(
                    "complete/fail 必须提供 claim_id 和结构化 result。",
                    success=False,
                )
            if action == "complete":
                required = (
                    "target",
                    "before_hash",
                    "after_hash",
                    "verification",
                )
            else:
                required = ("blocked_reason",)
            missing = [
                key
                for key in required
                if not str(result.get(key) or "").strip()
            ]
            if missing:
                return tool_error(
                    "result 缺少非空字段：" + ", ".join(missing),
                    success=False,
                )
            recorded = db.record_learning_candidate_result(
                candidate_id,
                version_hash=version,
                claim_id=claim_id,
                success=action == "complete",
                result=result,
            )
            if recorded is None:
                return tool_error(
                    "候选未获批准、版本已变化或状态不允许记录结果。", success=False
                )
            return json.dumps(
                {
                    "success": True,
                    "candidate": _candidate_view(requested_profile, recorded),
                    "application": result,
                },
                ensure_ascii=False,
            )
        reviewed = db.review_learning_candidate(
            candidate_id,
            action=action,
            version_hash=version,
        )
        if reviewed is None:
            return tool_error("候选版本已变化，请重新 list/get 后审核。", success=False)
        candidate = _candidate_view(requested_profile, reviewed)
        if action == "approve":
            candidate["next_action"] = (
                "这是用户在前台明确批准的候选。现在先核查目标所有权和现有实现；"
                "Skill 走目标 Profile 的 skill_manage/provenance 流程，Tool 走正常代码或"
                "配置修改及最小验证。不能修改时记录 blocked，禁止伪报完成。"
            )
        return json.dumps(
            {"success": True, "candidate": candidate}, ensure_ascii=False
        )
    finally:
        db.close()


LEARNING_REVIEW_SCHEMA = {
    "name": "learning_review",
    "description": (
        "审核 Episode 每三天归纳出的 Skill/Tool 候选。仅 Lingjun 的配置 owner 可用。"
        "list/get 读取有界候选；approve/reject 必须携带 Telegram 通知中的候选版本。"
        "批准不会由 cron 自动修改文件，而是授权当前前台 turn 按正常来源与验证流程处理。"
        "处理结束后必须用 complete/fail 写入目标、hash 和最小验证结果。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "get",
                    "approve",
                    "reject",
                    "claim",
                    "complete",
                    "fail",
                ],
                "description": "列出、查看、审核候选，或登记前台应用结果。",
            },
            "profile": {
                "type": "string",
                "description": "候选所属 Profile；审批时必填。",
            },
            "candidate_id": {
                "type": "integer",
                "description": "候选 ID；除 list 外必填。",
            },
            "version": {
                "type": "string",
                "description": "Telegram 摘要中的版本 hash；除 list/get 外必填。",
            },
            "result": {
                "type": "object",
                "description": (
                    "complete/fail 的结构化结果；应包含 target、before_hash、"
                    "after_hash、verification 或 blocked_reason。"
                ),
            },
            "claim_id": {
                "type": "string",
                "description": "claim 返回的租约 ID；complete/fail 必填。",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="learning_review",
    toolset="learning_review",
    schema=LEARNING_REVIEW_SCHEMA,
    handler=learning_review,
    emoji="🧭",
    check_fn=_available,
)
