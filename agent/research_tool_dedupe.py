"""Exact, session-scoped SmartSearch deduplication for research leaves."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


_SMART_SEARCH_PREFIX = "mcp__smart_search__"
_REQUEST_FIELDS = (("url", 480), ("query", 320), ("question", 320))
_FAILURE_STATUSES = frozenset(
    {"error", "failed", "failure", "timeout", "cancelled", "blocked"}
)
_DEFAULT_SINGLE_FLIGHT_WAIT_SECONDS = 180.0
_DEFAULT_MAX_INFLIGHT = 512
_DedupeKey = tuple[str, str, str, int]
_SMART_SEARCH_DEDUPE_TOOL_NAMES = frozenset(
    {
        f"{_SMART_SEARCH_PREFIX}smart_search",
        f"{_SMART_SEARCH_PREFIX}smart_fetch",
        f"{_SMART_SEARCH_PREFIX}smart_research",
        f"{_SMART_SEARCH_PREFIX}smart_map",
    }
)


def is_research_leaf_smart_search_dedupe_eligible(
    runtime_role: str,
    session_id: str,
    tool_name: str,
) -> bool:
    """Return whether a call is eligible for exact retrieval deduplication."""
    from agent.tool_result_classification import tool_may_have_side_effect

    return (
        runtime_role == "research_leaf"
        and bool(session_id)
        and tool_name in _SMART_SEARCH_DEDUPE_TOOL_NAMES
        and not tool_may_have_side_effect(tool_name)
    )


def _canonical_args(arguments: dict[str, Any]) -> str:
    return json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _bounded_text(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), default=str
        )
    text = re.sub(r"\s+", " ", text).strip()
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True)
    except Exception:
        pass
    if len(text) > limit:
        return text[: max(0, limit - 1)] + "…"
    return text


def request_ledger(arguments: Any) -> dict[str, str]:
    """Return the bounded retrieval coordinates safe to retain in receipts."""
    if not isinstance(arguments, dict):
        return {}
    ledger: dict[str, str] = {}
    for field, limit in _REQUEST_FIELDS:
        value = arguments.get(field)
        if value not in (None, ""):
            ledger[field] = _bounded_text(value, limit)
    return ledger


def _embedded_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        return value
    return None


def _successful_result(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        if (
            result.get("isError") is True
            or result.get("error") not in (None, "", False)
        ):
            return False
        if result.get("ok") is False or result.get("success") is False:
            return False
        if str(result.get("status") or "").strip().lower() in _FAILURE_STATUSES:
            return False
        nested = result.get("result")
        if isinstance(nested, (dict, list)):
            return _successful_result(nested)
        if isinstance(nested, str):
            parsed = _embedded_json(nested)
            if parsed is not None:
                return _successful_result(parsed)
        return True
    if isinstance(result, list):
        if any(
            isinstance(item, dict)
            and (item.get("isError") is True or item.get("error"))
            for item in result
        ):
            return False
        return bool(result)
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return False
        lowered = text.lower()
        if lowered.startswith(("error:", "error executing", "[tool execution cancelled")):
            return False
        parsed = _embedded_json(text)
        return _successful_result(parsed) if parsed is not None else True
    return True


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)


@dataclass
class _Flight:
    event: threading.Event
    started_at: float


@dataclass(frozen=True)
class _Success:
    chars: int
    sha256: str
    request: dict[str, str]


@dataclass(frozen=True)
class DedupeDecision:
    key: _DedupeKey | None = None
    flight: _Flight | None = None
    duplicate_result: str | None = None

    @property
    def owner(self) -> bool:
        return self.key is not None and self.flight is not None


class ResearchToolDeduper:
    """Bounded success cache plus single-flight coordination."""

    def __init__(
        self,
        *,
        max_entries: int = 512,
        max_entries_per_session: int = 128,
        single_flight_wait_seconds: float = _DEFAULT_SINGLE_FLIGHT_WAIT_SECONDS,
        single_flight_lease_seconds: float | None = None,
        max_inflight: int = _DEFAULT_MAX_INFLIGHT,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._max_entries = max(1, int(max_entries))
        self._max_entries_per_session = max(1, int(max_entries_per_session))
        self._single_flight_wait_seconds = max(
            0.01, float(single_flight_wait_seconds)
        )
        lease_seconds = (
            single_flight_wait_seconds
            if single_flight_lease_seconds is None
            else single_flight_lease_seconds
        )
        self._single_flight_lease_seconds = max(0.01, float(lease_seconds))
        self._max_inflight = max(1, int(max_inflight))
        self._clock = clock
        self._lock = threading.Lock()
        self._successes: OrderedDict[_DedupeKey, _Success] = OrderedDict()
        self._inflight: dict[_DedupeKey, _Flight] = {}

    @staticmethod
    def _eligible(runtime_role: str, session_id: str, tool_name: str) -> bool:
        return is_research_leaf_smart_search_dedupe_eligible(
            runtime_role,
            session_id,
            tool_name,
        )

    @staticmethod
    def _duplicate_receipt(
        tool_name: str,
        args_sha256: str,
        success: _Success,
    ) -> str:
        return json.dumps(
            {
                "ok": True,
                "kind": "smart_search_duplicate_receipt",
                "duplicate": True,
                "tool_name": tool_name,
                "canonical_args_sha256": args_sha256,
                "request": success.request,
                "original_result": {
                    "status": "success",
                    "chars": success.chars,
                    "sha256": success.sha256,
                },
                "next": (
                    "Use the prior result/evidence receipt; choose a new source "
                    "or summarize instead of repeating this exact call."
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _evict_stale_inflight_locked(self, now: float) -> None:
        stale = [
            (key, flight)
            for key, flight in self._inflight.items()
            if now - flight.started_at >= self._single_flight_lease_seconds
        ]
        for key, flight in stale:
            if self._inflight.get(key) is flight:
                self._inflight.pop(key, None)
                flight.event.set()

    def begin(
        self,
        *,
        runtime_role: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        tool_generation: int = 0,
    ) -> DedupeDecision:
        if not self._eligible(runtime_role, session_id, tool_name):
            return DedupeDecision()
        canonical = _canonical_args(arguments)
        args_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
        session_sha256 = hashlib.sha256(session_id.encode()).hexdigest()
        key = (
            session_sha256,
            tool_name,
            args_sha256,
            int(tool_generation),
        )
        while True:
            with self._lock:
                now = self._clock()
                self._evict_stale_inflight_locked(now)
                success = self._successes.get(key)
                if success is not None:
                    self._successes.move_to_end(key)
                    return DedupeDecision(
                        duplicate_result=self._duplicate_receipt(
                            tool_name, args_sha256, success
                        )
                    )
                flight = self._inflight.get(key)
                if flight is None:
                    if len(self._inflight) >= self._max_inflight:
                        # Coordination is bounded, but dedupe is fail-open:
                        # execute this read without tracking it rather than
                        # reject or delay a valid retrieval.
                        return DedupeDecision()
                    flight = _Flight(threading.Event(), started_at=now)
                    self._inflight[key] = flight
                    return DedupeDecision(key=key, flight=flight)
                lease_remaining = max(
                    0.01,
                    self._single_flight_lease_seconds
                    - (now - flight.started_at),
                )
            flight.event.wait(
                timeout=min(
                    self._single_flight_wait_seconds,
                    lease_remaining,
                )
            )

    def abort(self, decision: DedupeDecision) -> None:
        if not decision.owner:
            return
        with self._lock:
            flight = self._inflight.get(decision.key)
            if flight is decision.flight:
                self._inflight.pop(decision.key, None)
                flight.event.set()

    def finish(
        self,
        decision: DedupeDecision,
        result: Any,
        arguments: dict[str, Any],
    ) -> None:
        if not decision.owner:
            return
        try:
            text = _result_text(result)
            success = _successful_result(result)
            ledger = request_ledger(arguments)
            with self._lock:
                flight = self._inflight.get(decision.key)
                if flight is not decision.flight:
                    return
                try:
                    if success:
                        (
                            session_id,
                            _tool_name,
                            _args_sha256,
                            _generation,
                        ) = decision.key
                        self._successes[decision.key] = _Success(
                            chars=len(text),
                            sha256=hashlib.sha256(text.encode()).hexdigest(),
                            request=ledger,
                        )
                        self._successes.move_to_end(decision.key)
                        session_keys = [
                            key for key in self._successes if key[0] == session_id
                        ]
                        for key in session_keys[: -self._max_entries_per_session]:
                            self._successes.pop(key, None)
                        while len(self._successes) > self._max_entries:
                            self._successes.popitem(last=False)
                finally:
                    self._inflight.pop(decision.key, None)
                    flight.event.set()
        except BaseException:
            self.abort(decision)
            raise


_GLOBAL_DEDUPER = ResearchToolDeduper()


def begin_research_leaf_smart_search(
    *,
    runtime_role: str,
    session_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_generation: int,
) -> DedupeDecision:
    return _GLOBAL_DEDUPER.begin(
        runtime_role=runtime_role,
        session_id=session_id,
        tool_name=tool_name,
        arguments=arguments,
        tool_generation=tool_generation,
    )


def finish_research_leaf_smart_search(
    decision: DedupeDecision,
    result: Any,
    arguments: dict[str, Any],
) -> None:
    _GLOBAL_DEDUPER.finish(decision, result, arguments)


def abort_research_leaf_smart_search(decision: DedupeDecision) -> None:
    _GLOBAL_DEDUPER.abort(decision)
