"""Local Episode/scenario-card store with deterministic keyword retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_canonical_hermes_root


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    subject_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    body TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, run_id, source_hash)
);
CREATE TABLE IF NOT EXISTS scenario_cards (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    valid_until TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS scenario_keywords (
    card_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    normalized TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'phrase',
    PRIMARY KEY(card_id, normalized),
    FOREIGN KEY(card_id) REFERENCES scenario_cards(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    card_id TEXT NOT NULL,
    episode_id INTEGER,
    source_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY,
    event TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS session_injections (
    session_id TEXT NOT NULL,
    card_id TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    injected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(session_id, card_id, body_hash)
);
"""

_TOKEN_RE = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def subject_id_for_agent(agent: Any) -> str:
    source = str(getattr(agent, "platform", "") or "cli")
    user = str(
        getattr(agent, "_user_id_alt", "")
        or getattr(agent, "_user_id", "")
        or "local"
    )
    if source == "cli" and user == "local":
        source = "local"
        user = os.environ.get(
            "HERMES_EPISODE_LOCAL_USER_ID", "local-owner"
        ).strip()
    digest = hashlib.sha256(f"{source}\0{user}".encode()).hexdigest()[:24]
    return f"user-{digest}"


def profile_for_agent(agent: Any) -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


@dataclass(frozen=True)
class ScenarioHit:
    card_id: str
    title: str
    body: str
    body_hash: str
    score: int


class LocalContextStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else (
            get_canonical_hermes_root() / "user-context" / "context.db"
        )

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return conn

    def import_episode(
        self,
        *,
        subject_id: str,
        profile: str,
        run_id: str,
        source_hash: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO episodes "
                "(subject_id, profile, run_id, source_hash, body, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    subject_id,
                    profile,
                    run_id,
                    source_hash,
                    body,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
            return bool(cur.rowcount)

    def upsert_card(
        self,
        *,
        card_id: str,
        subject_id: str,
        profile: str,
        title: str,
        body: str,
        keywords: Iterable[str],
        status: str = "active",
        valid_until: str | None = None,
        allow_dynamic_infrastructure: bool = False,
    ) -> None:
        if not allow_dynamic_infrastructure:
            from agent.episode_policy import contains_dynamic_infrastructure

            if contains_dynamic_infrastructure(body):
                raise ValueError(
                    "dynamic infrastructure may remain Episode evidence but "
                    "cannot become a current-fact scenario card"
                )
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO scenario_cards "
                "(id, subject_id, profile, title, body, body_hash, status, valid_until) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title=excluded.title, body=excluded.body, "
                "body_hash=excluded.body_hash, status=excluded.status, "
                "valid_until=excluded.valid_until, updated_at=CURRENT_TIMESTAMP",
                (
                    card_id,
                    subject_id,
                    profile,
                    title,
                    body,
                    body_hash,
                    status,
                    valid_until,
                ),
            )
            conn.execute("DELETE FROM scenario_keywords WHERE card_id=?", (card_id,))
            for raw in keywords:
                normalized = normalize_text(str(raw))
                if normalized:
                    conn.execute(
                        "INSERT OR IGNORE INTO scenario_keywords "
                        "(card_id, keyword, normalized, kind) VALUES (?, ?, ?, ?)",
                        (
                            card_id,
                            str(raw),
                            normalized,
                            "phrase" if " " in normalized or len(normalized) >= 4 else "alias",
                        ),
                    )

    def search(
        self,
        query: str,
        *,
        subject_id: str,
        profile: str,
        session_id: str,
        limit: int = 3,
        char_cap: int = 3000,
    ) -> list[ScenarioHit]:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return []
        tokens = set(_TOKEN_RE.findall(normalized_query))
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.title, c.body, c.body_hash, k.normalized, k.kind "
                "FROM scenario_cards c JOIN scenario_keywords k ON k.card_id=c.id "
                "WHERE c.subject_id=? AND c.profile IN (?, '*') "
                "AND c.status='active' "
                "AND (c.valid_until IS NULL OR c.valid_until > CURRENT_TIMESTAMP)",
                (subject_id, profile),
            ).fetchall()
            scores: dict[str, tuple[sqlite3.Row, int]] = {}
            for row in rows:
                keyword = row["normalized"]
                score = 0
                if keyword and keyword in normalized_query:
                    score = 100 + min(len(keyword), 40)
                elif keyword in tokens:
                    score = 25
                if not score:
                    continue
                previous = scores.get(row["id"])
                if previous is None or score > previous[1]:
                    scores[row["id"]] = (row, score)
            hits: list[ScenarioHit] = []
            used = 0
            for row, score in sorted(
                scores.values(), key=lambda item: (-item[1], item[0]["id"])
            ):
                already = conn.execute(
                    "SELECT 1 FROM session_injections "
                    "WHERE session_id=? AND card_id=? AND body_hash=?",
                    (session_id, row["id"], row["body_hash"]),
                ).fetchone()
                if already:
                    continue
                body = str(row["body"])
                remaining = char_cap - used
                if remaining <= 0:
                    break
                body = body[:remaining]
                hits.append(
                    ScenarioHit(
                        row["id"], row["title"], body, row["body_hash"], score
                    )
                )
                used += len(body)
                conn.execute(
                    "INSERT OR IGNORE INTO session_injections "
                    "(session_id, card_id, body_hash) VALUES (?, ?, ?)",
                    (session_id, row["id"], row["body_hash"]),
                )
                if len(hits) >= limit:
                    break
            return hits


def scenario_context_for_turn(agent: Any, query: str) -> str:
    if getattr(agent, "runtime_role", "") == "research_leaf":
        return ""
    try:
        hits = LocalContextStore().search(
            query,
            subject_id=subject_id_for_agent(agent),
            profile=profile_for_agent(agent),
            session_id=str(getattr(agent, "session_id", "") or "unsaved"),
        )
    except Exception:
        return ""
    if not hits:
        return ""
    blocks = [
        f"[情景知识 {hit.card_id}] {hit.title}\n{hit.body}"
        for hit in hits
    ]
    return "仅在与当前请求相关时参考以下本地情景知识：\n\n" + "\n\n".join(blocks)


def store_extracted_episode(item: dict[str, Any]) -> bool:
    """Store one validated Episode extractor output in the canonical DB."""
    required = ("subject_id", "profile", "source_hash", "session_id")
    missing = [key for key in required if not str(item.get(key) or "").strip()]
    if missing:
        raise ValueError(f"episode missing required fields: {', '.join(missing)}")
    body_fields = (
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
    )
    body = json.dumps(
        {key: item.get(key) for key in body_fields},
        ensure_ascii=False,
        sort_keys=True,
    )
    return LocalContextStore().import_episode(
        subject_id=str(item["subject_id"]),
        profile=str(item["profile"]),
        run_id=(
            f"hermes:{item['subject_id']}:{item['profile']}:{item['session_id']}"
        ),
        source_hash=str(item["source_hash"]),
        body=body,
        metadata={
            "memory_kind": "episode",
            "source_session_id": item["session_id"],
            "outcome": item.get("outcome"),
        },
    )
