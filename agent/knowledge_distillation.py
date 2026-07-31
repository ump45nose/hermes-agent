"""Daily Episode-driven Markdown distillation and learning candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

MEMORY_MAX_INPUT_CHARS = 192_000
MEMORY_MAX_OUTPUT_TOKENS = 4_000
CANDIDATE_MAX_OUTPUT_TOKENS = 4_000
DISTILLER_VERSION = 1
_JOURNAL_NAME = ".episode-distillation-journal.json"

_MEMORY_SYSTEM = """\
你是 Hermes 的长期知识归纳器。输入包含同一个 Profile 的新 Episode、当前
MEMORY.md/USER.md 条目和允许写 USER.md 的 owner subject。
只输出 JSON object：
{"operations":[{"target":"memory|user","action":"add|replace|remove|skip",
"content":"","old_text":"","evidence_episode_ids":["<输入中的真实 Episode id>"],
"reason":""}]}

memory 只保存稳定、可复用的工作方式、约定和经验；不要保存动态基础设施状态、
任务流水、一次性错误、群成员个人资料或凭据。user 只保存 owner subject 的稳定
事实、偏好和沟通方式。没有价值时使用 skip。replace/remove 只能针对输入中
标记 auto_managed=true 的完整条目。不得改写 manual 条目。
每个 operation 的 evidence_episode_ids 必须直接复制
valid_evidence_episode_ids 中至少一个真实整数；禁止照抄示例、猜测或虚构 ID。
若所有 Episode 都没有长期价值，返回 {"operations":[]}。"""

_CANDIDATE_SYSTEM = """\
你是 Hermes 的 Skill/Tool 候选归纳器。输入只来自同一个 Profile 的 Episode，
包含确定性 Tool receipt 摘要以及现有候选。先合并近义候选，不维护手写同义词表。
只输出 JSON object：
{"proposals":[
{"kind":"skill","semantic_key":"stable-kebab-case","title":"",
"payload":{"trigger":"","procedure":[],"verification":[],"anti_conditions":[]},
"evidence":[{"episode_id":1,"evidence_count":1}]},
{"kind":"tool","semantic_key":"tool-name-issue","title":"",
"payload":{"tool_name":"","ownership":"unknown","issue_type":"","symptom":"",
"expected_behavior":"","reproduction":"","change_scope":""},
"evidence":[{"episode_id":2,"evidence_count":3}]}
],"discarded_episode_ids":[]}

Tool 名只能来自 tool_evidence.tool_name，不得输出 <unknown> 或猜测。单条偶发失败
不要生成 Tool 候选。Skill 必须是可复用流程，不是一次任务叙事。"""


@contextmanager
def _profile_home(path: Path):
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous


def _decode_object(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("knowledge distiller did not return a JSON object")


def _response_text(response: Any) -> str:
    message = response.choices[0].message
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", message) or "")


def _call(system: str, payload: Dict[str, Any], *, task: str) -> Dict[str, Any]:
    from agent.auxiliary_client import call_llm

    response = call_llm(
        task=task,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                )[:MEMORY_MAX_INPUT_CHARS],
            },
        ],
        max_tokens=(
            MEMORY_MAX_OUTPUT_TOKENS
            if task == "knowledge_distillation"
            else CANDIDATE_MAX_OUTPUT_TOKENS
        ),
        temperature=0,
        timeout=90.0,
    )
    return _decode_object(_response_text(response))


def _episode_view(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = json.loads(row.get("payload_json") or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "id": int(row["id"]),
        "subject_id": str(row.get("subject_id") or ""),
        "source_hash": str(row.get("source_hash") or ""),
        "outcome": str(row.get("outcome") or "unknown"),
        "title": str(row.get("title") or ""),
        "summary": str(row.get("summary") or ""),
        "payload": payload,
        "tool_evidence": row.get("tool_evidence") or [],
    }


def _hash_entries(entries: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(entries), ensure_ascii=False).encode()
    ).hexdigest()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    from utils import atomic_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".episode-distillation-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        atomic_replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _recover_memory_journal(store: Any, profile_home: Path) -> None:
    journal_path = profile_home / "memories" / _JOURNAL_NAME
    if not journal_path.exists():
        return
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        before = journal["before"]
        after = journal["after"]
        if not all(
            isinstance(value, list)
            for value in (
                before.get("memory"),
                before.get("user"),
                after.get("memory"),
                after.get("user"),
            )
        ):
            raise ValueError("invalid memory journal shape")
    except Exception as exc:
        raise RuntimeError(f"cannot recover memory journal: {exc}") from exc

    with ExitStack() as locks:
        for target in ("memory", "user"):
            locks.enter_context(store._file_lock(store._path_for(target)))
        for target in ("memory", "user"):
            reloaded = store._reload_target(target, skip_drift=True)
            if reloaded is not None:
                raise RuntimeError(
                    f"cannot read {target} while recovering memory journal"
                )
        current = {
            "memory": list(store.memory_entries),
            "user": list(store.user_entries),
        }
        if current == after or current == before:
            journal_path.unlink()
            return
        partial = all(
            current[target] in (before[target], after[target])
            for target in ("memory", "user")
        )
        if not partial:
            raise RuntimeError(
                "memory files drifted after an interrupted distillation; "
                f"journal retained at {journal_path}"
            )
        for target in ("memory", "user"):
            store._write_file(store._path_for(target), before[target])
            store._set_entries(target, list(before[target]))
        journal_path.unlink()


def distill_profile_memory(
    db: Any,
    *,
    profile: str,
    profile_home: Path,
    owner_subjects: Iterable[str],
    limit: int = 50,
) -> Dict[str, Any]:
    episodes = db.list_episodes_for_distillation(
        profile=profile, limit=limit, candidate_kinds=False
    )
    if not episodes:
        return {"status": "none", "processed": 0, "writes": 0}
    claim = db.claim_knowledge_distillation(
        profile=profile,
        episode_ids=[row["id"] for row in episodes],
        extractor_model="knowledge_distillation",
    )
    if claim is None:
        return {"status": "none", "processed": 0, "writes": 0}
    try:
        from tools.memory_tool import MemoryStore

        with _profile_home(profile_home):
            store = MemoryStore(memory_char_limit=4_000, user_char_limit=3_000)
            store.load_from_disk()
            _recover_memory_journal(store, profile_home)
            memory_entries = list(store.memory_entries)
            user_entries = list(store.user_entries)
            managed = {
                entry: db.is_auto_managed_memory_entry(entry)
                for entry in memory_entries + user_entries
            }
            result = _call(
                _MEMORY_SYSTEM,
                {
                    "profile": profile,
                    "owner_subjects": sorted(set(owner_subjects)),
                    "valid_evidence_episode_ids": [
                        int(row["id"]) for row in episodes
                    ],
                    "memory_entries": [
                        {"text": value, "auto_managed": managed[value]}
                        for value in memory_entries
                    ],
                    "user_entries": [
                        {"text": value, "auto_managed": managed[value]}
                        for value in user_entries
                    ],
                    "episodes": [_episode_view(row) for row in episodes],
                },
                task="knowledge_distillation",
            )
            operations = result.get("operations")
            if not isinstance(operations, list):
                raise ValueError("operations must be a list")
            allowed_ids = {int(row["id"]) for row in episodes}
            owners = set(owner_subjects)
            episode_by_id = {int(row["id"]): row for row in episodes}
            writes = 0
            applied: Dict[str, set[int]] = {"memory": set(), "user": set()}
            prepared: Dict[str, list[Dict[str, Any]]] = {
                "memory": [],
                "user": [],
            }
            audit_specs: list[Dict[str, Any]] = []
            for operation in operations:
                if not isinstance(operation, dict):
                    raise ValueError("operation must be an object")
                target = str(operation.get("target") or "")
                action = str(operation.get("action") or "")
                if target not in {"memory", "user"}:
                    raise ValueError("invalid memory target")
                evidence_ids = {
                    int(value)
                    for value in (operation.get("evidence_episode_ids") or [])
                    if int(value) in allowed_ids
                }
                if not evidence_ids:
                    raise ValueError("operation has no valid Episode evidence")
                if target == "user":
                    evidence_ids = {
                        value
                        for value in evidence_ids
                        if str(episode_by_id[value].get("subject_id") or "") in owners
                    }
                    if not evidence_ids:
                        continue
                if action == "skip":
                    continue
                applied[target].update(evidence_ids)
                old_text = str(operation.get("old_text") or "").strip()
                content = str(operation.get("content") or "").strip()
                if action not in {"add", "replace", "remove"}:
                    raise ValueError("invalid memory action")
                if action in {"add", "replace"}:
                    from tools.memory_tool import validate_memory_content

                    content_error = validate_memory_content(target, content)
                    if content_error:
                        raise ValueError(content_error)
                if action in {"replace", "remove"}:
                    entries = (
                        store.user_entries if target == "user" else store.memory_entries
                    )
                    matches = [entry for entry in entries if old_text and old_text in entry]
                    if len(matches) != 1 or not db.is_auto_managed_memory_entry(matches[0]):
                        raise ValueError("automatic write attempted to modify manual entry")
                op = {"action": action}
                if content:
                    op["content"] = content
                if old_text:
                    op["old_text"] = old_text
                prepared[target].append(op)
                new_entry = content if action in {"add", "replace"} else ""
                audit_specs.append(
                    {
                        "target": target,
                        "action": action,
                        "content_hash": (
                        hashlib.sha256(new_entry.encode()).hexdigest()
                        if new_entry
                        else hashlib.sha256(matches[0].encode()).hexdigest()
                        ),
                        "episode_ids": evidence_ids,
                    }
                )

            # Validate every operation and final budget in memory before the
            # first file mutation. This prevents malformed JSON or a later
            # over-budget operation from leaving the other Markdown file
            # partially updated.
            simulated: Dict[str, list[str]] = {
                "memory": list(store.memory_entries),
                "user": list(store.user_entries),
            }
            limits = {"memory": 4_000, "user": 3_000}
            for target, target_ops in prepared.items():
                working = simulated[target]
                for op in target_ops:
                    action = op["action"]
                    content = str(op.get("content") or "").strip()
                    old_text = str(op.get("old_text") or "").strip()
                    if action == "add":
                        if not content:
                            raise ValueError("add requires content")
                        if content not in working:
                            working.append(content)
                    else:
                        matches = [
                            index for index, entry in enumerate(working)
                            if old_text and old_text in entry
                        ]
                        if len(matches) != 1:
                            raise ValueError("replace/remove must match one entry")
                        if action == "replace":
                            if not content:
                                raise ValueError("replace requires content")
                            working[matches[0]] = content
                        else:
                            working.pop(matches[0])
                if len("\n§\n".join(working)) > limits[target]:
                    raise ValueError(f"{target} final content exceeds configured limit")

            originals = {
                "memory": list(store.memory_entries),
                "user": list(store.user_entries),
            }
            journal_path = profile_home / "memories" / _JOURNAL_NAME
            if any(prepared.values()):
                with ExitStack() as locks:
                    for target in ("memory", "user"):
                        locks.enter_context(
                            store._file_lock(store._path_for(target))
                        )
                    for target in ("memory", "user"):
                        reloaded = store._reload_target(target)
                        if reloaded is not None:
                            raise RuntimeError(
                                f"{target} changed before atomic distillation commit"
                            )
                        if store._entries_for(target) != originals[target]:
                            raise RuntimeError(
                                f"{target} changed before atomic distillation commit"
                            )
                    _write_json_atomic(
                        journal_path,
                        {
                            "version": 1,
                            "profile": profile,
                            "claim_id": int(claim["id"]),
                            "before": originals,
                            "after": simulated,
                        },
                    )
                    try:
                        for target in ("memory", "user"):
                            if not prepared[target]:
                                continue
                            store._write_file(
                                store._path_for(target),
                                simulated[target],
                            )
                            store._set_entries(
                                target,
                                list(simulated[target]),
                            )
                    except Exception:
                        for target in ("memory", "user"):
                            store._write_file(
                                store._path_for(target),
                                originals[target],
                            )
                            store._set_entries(
                                target,
                                list(originals[target]),
                            )
                        raise
                journal_path.unlink()

            before_hashes = {
                target: _hash_entries(originals[target])
                for target in ("memory", "user")
            }
            after_hashes = {
                target: _hash_entries(
                    store.user_entries if target == "user" else store.memory_entries
                )
                for target in ("memory", "user")
            }
            for audit in audit_specs:
                db.record_memory_audit(
                    target=audit["target"],
                    action=audit["action"],
                    origin="episode_distillation",
                    before_hash=before_hashes[audit["target"]],
                    after_hash=after_hashes[audit["target"]],
                    content_hash=audit["content_hash"],
                    episode_ids=audit["episode_ids"],
                )
                writes += 1
            for row in episodes:
                episode_id = int(row["id"])
                db.record_knowledge_disposition(
                    episode_id,
                    kind="memory",
                    disposition="applied" if episode_id in applied["memory"] else "skipped",
                )
                if str(row.get("subject_id") or "") not in owners:
                    user_disposition = "excluded_by_policy"
                else:
                    user_disposition = (
                        "applied" if episode_id in applied["user"] else "skipped"
                    )
                db.record_knowledge_disposition(
                    episode_id, kind="user", disposition=user_disposition
                )
        db.finish_knowledge_distillation(claim["id"], success=True)
        return {
            "status": "succeeded",
            "processed": len(episodes),
            "writes": writes,
        }
    except Exception as exc:
        db.finish_knowledge_distillation(
            claim["id"], success=False, error=str(exc)
        )
        return {
            "status": "failed",
            "processed": len(episodes),
            "writes": 0,
            "error": str(exc),
        }


def consolidate_profile_candidates(
    db: Any,
    *,
    profile: str,
    limit: int = 50,
) -> Dict[str, Any]:
    episodes = db.list_episodes_for_distillation(
        profile=profile, limit=limit, candidate_kinds=True
    )
    if not episodes:
        return {"status": "none", "processed": 0, "updates": 0}
    existing = db.search_learning_candidates(
        " ".join(
            str(row.get("title") or "") + " " + str(row.get("retrieval_text") or "")
            for row in episodes
        ),
        limit=50,
    )
    try:
        result = _call(
            _CANDIDATE_SYSTEM,
            {
                "profile": profile,
                "episodes": [_episode_view(row) for row in episodes],
                "existing_candidates": [
                    {
                        "id": int(row["id"]),
                        "kind": row["kind"],
                        "semantic_key": row["semantic_key"],
                        "title": row["title"],
                        "payload": json.loads(row["payload_json"]),
                    }
                    for row in existing
                ],
            },
            task="learning_candidate_distillation",
        )
        proposals = result.get("proposals", result.get("candidates"))
        if proposals is None and (
            result.get("skip") is True
            or isinstance(result.get("discarded_episode_ids"), list)
        ):
            proposals = []
        if not isinstance(proposals, list):
            raise ValueError("proposals must be a list")
        by_id = {int(row["id"]): row for row in episodes}
        updates = 0
        assigned: Dict[str, set[int]] = {"skill": set(), "tool": set()}
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            kind = str(proposal.get("kind") or "")
            key = str(proposal.get("semantic_key") or "").strip()
            if kind not in {"skill", "tool"} or not re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*", key
            ):
                raise ValueError("invalid candidate kind or semantic_key")
            evidence = []
            for item in proposal.get("evidence") or []:
                episode_id = int(item.get("episode_id") or 0)
                if episode_id not in by_id:
                    continue
                row = by_id[episode_id]
                if kind == "tool":
                    deterministic = [
                        value
                        for value in row.get("tool_evidence") or []
                        if str(value.get("tool_name") or "")
                    ]
                    tool_names = {
                        str(value.get("tool_name") or "") for value in deterministic
                    }
                    payload_tool = str(
                        (proposal.get("payload") or {}).get("tool_name") or ""
                    )
                    if not payload_tool or payload_tool not in tool_names:
                        raise ValueError("tool candidate name lacks deterministic evidence")
                    failure_statuses = {
                        "error", "failed", "failure", "timeout", "denied", "blocked"
                    }
                    evidence_count = sum(
                        1
                        for value in deterministic
                        if str(value.get("tool_name") or "") == payload_tool
                        and str(value.get("result_status") or "").lower()
                        in failure_statuses
                    )
                    if evidence_count == 0:
                        # Two successful/unknown receipts are not two
                        # independent failures. Keep Tool readiness grounded
                        # in deterministic failure statuses only.
                        continue
                else:
                    evidence_count = 1
                evidence.append(
                    {
                        "episode_id": episode_id,
                        "source_hash": row["source_hash"],
                        "outcome": row.get("outcome") or "",
                        "evidence_count": max(1, evidence_count),
                    }
                )
                assigned[kind].add(episode_id)
            if not evidence:
                continue
            db.upsert_learning_candidate(
                kind=kind,
                semantic_key=key,
                title=str(proposal.get("title") or key),
                payload=proposal.get("payload") or {},
                evidence=evidence,
            )
            updates += 1
        for row in episodes:
            episode_id = int(row["id"])
            for kind in ("skill", "tool"):
                db.record_knowledge_disposition(
                    episode_id,
                    kind=kind,
                    disposition=(
                        "candidate" if episode_id in assigned[kind] else "discarded"
                    ),
                )
        return {
            "status": "succeeded",
            "processed": len(episodes),
            "updates": updates,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "processed": len(episodes),
            "updates": 0,
            "error": str(exc),
        }
