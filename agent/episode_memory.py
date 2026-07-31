"""Profile-local Episode extraction and transient recall."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = 1
EXTRACTOR_MAX_INPUT_CHARS = 96_000
EXTRACTOR_MAX_OUTPUT_TOKENS = 1_200
EXTRACTOR_TIMEOUT_SECONDS = 60.0
RECALL_LIMIT = 3
RECALL_CHAR_CAP = 3_000

_EPISODE_FIELDS = (
    "title",
    "goal",
    "context",
    "actions",
    "decisions",
    "outcome",
    "summary",
    "artifacts",
    "open_loops",
    "reusable_lesson",
    "retrieval_text",
    "keywords",
)

_EXTRACTION_SYSTEM_PROMPT = """\
你是 Hermes 的 Episode 提炼器。输入只包含一个已经完成的 user turn：
用户/助手正文，以及脱敏后的确定性工具回执。

只输出一个 JSON object，不要 Markdown，不要推理过程。
有可复用的目标、决策、行动、结果、未完成事项或经验时：
{"kind":"episode","title":"","goal":"","context":"","actions":[],
"decisions":[],"outcome":"success|partial|failed|abandoned|unknown",
"summary":"","artifacts":[],"open_loops":[],"reusable_lesson":"",
"retrieval_text":"","keywords":[]}

只是寒暄、噪声、重复确认、无结果的测试时：
{"kind":"skip","reason":"简短原因"}

不得补写输入中没有的事实；不得保留凭据、token、system prompt、reasoning、
原始用户 ID 或原始 Tool result。retrieval_text 用于未来检索，应短而具体；
keywords 只给明确主题、对象、工具或决策短语。"""


def subject_id(source: str, user_id: str) -> str:
    source_value = str(source or "local")
    user_value = str(user_id or "local-owner")
    digest = hashlib.sha256(
        f"{source_value}\0{user_value}".encode()
    ).hexdigest()[:24]
    return f"user-{digest}"


def subject_id_for_agent(agent: Any) -> str:
    source = str(getattr(agent, "platform", "") or "cli")
    user = str(
        getattr(agent, "_user_id_alt", "")
        or getattr(agent, "_user_id", "")
        or "local"
    )
    if source == "cli" and user == "local":
        source = "local"
        user = "local-owner"
    return subject_id(source, user)


def profile_for_agent(agent: Any) -> str:
    configured = str(getattr(agent, "_episode_memory_profile", "") or "")
    if configured:
        return configured
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _response_text(response: Any) -> str:
    message = response.choices[0].message
    if isinstance(message, dict):
        value = message.get("content")
    else:
        value = getattr(message, "content", message)
    return str(value or "").strip()


def _decode_json_object(text: str) -> Dict[str, Any]:
    try:
        from agent.agent_runtime_helpers import strip_think_blocks

        text = strip_think_blocks(None, text).strip()
    except Exception:
        pass
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
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
    raise ValueError("episode extractor did not return a JSON object")


def _validate_payload(value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = str(value.get("kind") or "").lower()
    if kind == "skip":
        return None
    if kind != "episode":
        raise ValueError("episode extractor kind must be episode or skip")
    payload = {key: value.get(key) for key in _EPISODE_FIELDS}
    if not str(payload.get("summary") or "").strip():
        raise ValueError("episode summary is required")
    if not str(payload.get("retrieval_text") or "").strip():
        payload["retrieval_text"] = payload["summary"]
    if not isinstance(payload.get("keywords"), list):
        raise ValueError("episode keywords must be a list")
    return payload


def extract_one_candidate(
    db: Any,
    *,
    session_id: str,
    profile: str,
    subject: str,
    trigger_source: str,
    retry_failed: bool = False,
    retry_failed_before: Optional[float] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    extractor_model: str = "",
) -> Dict[str, Any]:
    """Claim and synchronously extract one complete turn."""
    candidate = db.claim_episode_candidate(
        session_id,
        trigger_source=trigger_source,
        max_chars=EXTRACTOR_MAX_INPUT_CHARS,
        retry_failed=retry_failed,
        retry_failed_before=retry_failed_before,
    )
    if candidate is None:
        return {"status": "none"}
    extraction_id = int(candidate["extraction_id"])
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task="episode_extraction",
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        candidate["messages"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            max_tokens=EXTRACTOR_MAX_OUTPUT_TOKENS,
            temperature=0,
            timeout=EXTRACTOR_TIMEOUT_SECONDS,
            main_runtime=main_runtime,
        )
        payload = _validate_payload(_decode_json_object(_response_text(response)))
        episode_id = db.complete_episode_extraction(
            extraction_id,
            subject_id=subject,
            profile=profile,
            payload=payload,
            extractor_model=extractor_model,
        )
        return {
            "status": "skipped" if payload is None else "succeeded",
            "episode_id": episode_id,
            "extraction_id": extraction_id,
        }
    except Exception as exc:
        db.fail_episode_extraction(extraction_id, str(exc))
        logger.warning(
            "Episode extraction failed open session=%s extraction=%s: %s",
            session_id,
            extraction_id,
            exc,
        )
        return {
            "status": "failed",
            "extraction_id": extraction_id,
            "error": str(exc),
        }


def maybe_extract_before_projection(
    agent: Any,
    *,
    estimated_input_tokens: int,
    effective_trigger: int,
) -> Dict[str, Any]:
    """Run at most one synchronous extraction per user turn."""
    if not getattr(agent, "_episode_memory_enabled", True):
        return {"status": "disabled"}
    if estimated_input_tokens <= effective_trigger:
        return {"status": "below_trigger"}
    turn = int(getattr(agent, "_user_turn_count", 0) or 0)
    if getattr(agent, "_episode_extraction_attempted_turn", None) == turn:
        return {"status": "already_attempted"}
    agent._episode_extraction_attempted_turn = turn
    db = getattr(agent, "_session_db", None)
    session_id_value = str(getattr(agent, "session_id", "") or "")
    if db is None or not session_id_value:
        return {"status": "unavailable"}
    try:
        session = db.get_session(session_id_value) or {}
        source = str(session.get("source") or getattr(agent, "platform", "") or "local")
        user = str(
            session.get("user_id")
            or getattr(agent, "_user_id_alt", "")
            or getattr(agent, "_user_id", "")
            or "local-owner"
        )
        runtime = (
            agent._current_main_runtime()
            if callable(getattr(agent, "_current_main_runtime", None))
            else None
        )
        return extract_one_candidate(
            db,
            session_id=session_id_value,
            profile=profile_for_agent(agent),
            subject=subject_id(source, user),
            trigger_source="context_trigger",
            main_runtime=runtime,
            extractor_model=str(getattr(agent, "model", "") or ""),
        )
    except Exception as exc:
        logger.warning("Episode trigger failed open: %s", exc)
        return {"status": "failed", "error": str(exc)}


def _allowed_scopes(
    agent: Any,
    *,
    current_subject: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    current = profile_for_agent(agent)
    configured = getattr(agent, "_episode_memory_read_scopes", ())
    values = []
    for profile_value, subject_values in configured or (
        ("self", ("self",)),
    ):
        profile = current if str(profile_value) == "self" else str(profile_value)
        subjects = []
        for subject_value in subject_values:
            subject = (
                current_subject
                if str(subject_value) == "self"
                else str(subject_value)
            )
            if re.fullmatch(r"user-[0-9a-f]{24}", subject) and subject not in subjects:
                subjects.append(subject)
        if profile and subjects:
            scope = (profile, tuple(subjects))
            if scope not in values:
                values.append(scope)
    return tuple(values or ((current, (current_subject,)),))


def _state_db_path(profile: str, current_db: Any) -> Path:
    current_path = Path(getattr(current_db, "db_path", "")).resolve()
    current_home = current_path.parent
    root = (
        current_home.parents[1]
        if current_home.parent.name == "profiles"
        else current_home
    )
    return (
        root / "state.db"
        if profile == "default"
        else root / "profiles" / profile / "state.db"
    )


def episode_context_for_turn(agent: Any, query: str) -> str:
    """Recall Episodes into this provider request only."""
    agent._pending_episode_injections = ()
    if not getattr(agent, "_episode_memory_enabled", True):
        return ""
    if getattr(agent, "runtime_role", "") == "research_leaf":
        return ""
    normalized = str(query or "").strip()
    if not normalized:
        return ""
    current_profile = profile_for_agent(agent)
    current_db = getattr(agent, "_session_db", None)
    subject = subject_id_for_agent(agent)
    candidates = []
    opened = []
    try:
        from hermes_state import SessionDB

        for profile, subjects in _allowed_scopes(
            agent,
            current_subject=subject,
        ):
            if profile == current_profile and current_db is not None:
                db = current_db
            else:
                path = _state_db_path(profile, current_db)
                if not path.exists():
                    continue
                db = SessionDB(db_path=path, read_only=True)
                opened.append(db)
            for allowed_subject in subjects:
                try:
                    hits = db.search_episodes(
                        normalized,
                        subject_id=allowed_subject,
                        profile=profile,
                        limit=RECALL_LIMIT,
                    )
                except Exception:
                    continue
                for hit in hits:
                    hit["_source_profile"] = profile
                    hit["_source_subject"] = allowed_subject
                    candidates.append(hit)
    finally:
        for db in opened:
            try:
                db.close()
            except Exception:
                pass
    candidates.sort(
        key=lambda item: (
            float(item.get("rank") or 1000.0),
            -float(item.get("created_at") or 0.0),
        )
    )
    selected = []
    selected_keys = set()
    used = 0
    blocks = []
    for item in candidates:
        if len(selected) >= RECALL_LIMIT:
            break
        item_key = (str(item["_source_profile"]), int(item["id"]))
        if item_key in selected_keys:
            continue
        body = str(item.get("retrieval_text") or item.get("summary") or "").strip()
        if not body:
            continue
        remaining = RECALL_CHAR_CAP - used
        if remaining <= 0:
            break
        body = body[:remaining]
        profile = str(item["_source_profile"])
        blocks.append(
            f"[Episode {profile}:{item['id']}] {item['title']}\n{body}"
        )
        selected.append(
            {
                "profile": profile,
                "id": item["id"],
                "body_hash": item["body_hash"],
                "score": item.get("rank") or 0.0,
            }
        )
        selected_keys.add(item_key)
        used += len(body)
    if not blocks:
        agent._pending_episode_injections = ()
        return ""
    agent._pending_episode_injections = tuple(selected)
    return (
        "仅在与当前请求相关时参考以下历史 Episode；它们不是当前环境事实：\n\n"
        + "\n\n".join(blocks)
    )
