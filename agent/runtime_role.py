"""Explicit process-role resolution for prompt overlays and capabilities."""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


RUNTIME_ROLES = frozenset(
    {
        "interactive",
        "kanban_controller",
        "kanban_worker",
        "cron",
        "subagent",
        "research_leaf",
    }
)

KANBAN_WORKER_OVERLAY = (
    "无人值守 worker：先读取当前任务；执行期间按需 heartbeat；"
    "只处理当前 task/run。完成时提交 complete，无法继续时提交 block，"
    "并附上结果、证据和未解决项。不得创建或分发新任务。"
)

CRON_OVERLAY = "这是定时进程：只执行当前计划任务，保持幂等，并明确记录本次实际结果。"

RESEARCH_LEAF_PROMPT = (
    "你是独立研究 leaf。使用可用研究工具深入检索并保存完整 Evidence bundle。"
    "最终只返回一个合法 JSON 对象（不要 Markdown 围栏），字段为 claims、source_ids、"
    "contradictions、unexpected_findings、unresolved；运行时会补充 artifact 路径和 "
    "SHA-256。不得分发、修改 Kanban、写 memory/shared-state。"
)

TOOL_ARTIFACT_TOOLSET = "tool_artifact"


@dataclass(frozen=True)
class RuntimeRoleResolution:
    role: str
    verified: bool
    reason: str


def runtime_capability_overlay(
    role: str,
    *,
    direct: set[str] | frozenset[str],
    deferred: set[str] | frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Apply process-identity-only capability overlays."""
    resolved_direct = set(direct)
    resolved_deferred = set(deferred)
    if role == "research_leaf":
        resolved_direct.add(TOOL_ARTIFACT_TOOLSET)
        resolved_deferred.discard(TOOL_ARTIFACT_TOOLSET)
    else:
        resolved_direct.discard(TOOL_ARTIFACT_TOOLSET)
        resolved_deferred.discard(TOOL_ARTIFACT_TOOLSET)
    return frozenset(resolved_direct), frozenset(resolved_deferred)


def scope_runtime_toolsets(
    role: str,
    *,
    enabled: list[str] | None,
    disabled: list[str] | None,
) -> tuple[list[str] | None, list[str]]:
    """Enforce role-scoped toolsets before legacy/default-all resolution."""
    scoped_disabled = list(disabled or [])
    if role == "research_leaf":
        scoped_enabled = None if enabled is None else list(enabled)
        scoped_disabled = [
            name for name in scoped_disabled if name != TOOL_ARTIFACT_TOOLSET
        ]
    else:
        scoped_enabled = (
            None
            if enabled is None
            else [name for name in enabled if name != TOOL_ARTIFACT_TOOLSET]
        )
        if TOOL_ARTIFACT_TOOLSET not in scoped_disabled:
            scoped_disabled.append(TOOL_ARTIFACT_TOOLSET)
    return scoped_enabled, scoped_disabled


def _verified_worker_from_env() -> RuntimeRoleResolution:
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "").strip()
    db_path = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if not all((task_id, run_id, claim_lock, db_path)):
        return RuntimeRoleResolution(
            "interactive", False, "worker markers incomplete"
        )
    path = Path(db_path)
    if not path.is_file():
        return RuntimeRoleResolution("interactive", False, "kanban database missing")
    try:
        expected_run = int(run_id)
    except ValueError:
        return RuntimeRoleResolution("interactive", False, "invalid run id")
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT status, current_run_id, claim_lock, claim_expires "
                "FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        return RuntimeRoleResolution("interactive", False, f"lease read failed: {exc}")
    if row is None:
        return RuntimeRoleResolution("interactive", False, "task not found")
    valid = (
        row["status"] == "running"
        and int(row["current_run_id"] or 0) == expected_run
        and str(row["claim_lock"] or "") == claim_lock
        and float(row["claim_expires"] or 0) > time.time()
    )
    if not valid:
        return RuntimeRoleResolution("interactive", False, "lease validation failed")
    return RuntimeRoleResolution("kanban_worker", True, "dispatcher lease verified")


def resolve_runtime_role(
    requested: str | None,
    *,
    platform: str | None = None,
) -> RuntimeRoleResolution:
    """Resolve a trusted runtime role, failing closed on conflicts."""
    requested_role = str(requested or "").strip()
    worker = _verified_worker_from_env()
    if worker.verified:
        if requested_role and requested_role not in {"kanban_worker", "interactive"}:
            return RuntimeRoleResolution(
                "interactive", False, "requested role conflicts with worker lease"
            )
        return worker
    if requested_role == "kanban_worker":
        return worker
    if requested_role:
        if requested_role not in RUNTIME_ROLES:
            return RuntimeRoleResolution("interactive", False, "unknown runtime role")
        return RuntimeRoleResolution(requested_role, True, "explicit process role")
    if platform == "subagent":
        return RuntimeRoleResolution("subagent", True, "subagent platform")
    return RuntimeRoleResolution("interactive", True, "default interactive role")
