"""Progressive tool disclosure ("tool search") for Hermes Agent.

The model sees a small eager tool surface plus ``tool_search``. Search results
carry structured tool references; the agent harness hydrates those references
into real tool schemas on the next model request. The model then calls the real
tool directly. ``tool_describe`` and ``tool_call`` remain executable only for
old conversation compatibility and are no longer advertised to new turns.

Design constraints this module is built around (see ``openclaw-tool-search-report``
for the full rationale):

* Core tools defined in ``toolsets._HERMES_CORE_TOOLS`` are *never* deferred.
  Always-load means always-load. No exceptions.
* Session-gated GUI toolsets (``desktop_ui``, ``project``) are also never
  deferred. They stay off the core list so CLI and messaging never pay for
  their schemas, but once a session enables them they stay in the
  model-facing array. Tool Search is for MCP/plugin catalog bloat, not for
  hiding the tools that define this session's surface.
* Tiered disclosure (July 2026 plan): the moment ANY deferrable (MCP/plugin)
  tools are present, they hide behind the bridge. What scales with catalog
  size is the *listing*, not the activation decision:
    - Tier 0 — no MCP/plugin tools: pure passthrough, everything eager.
    - Tier 1 — deferred tools whose catalog listing fits the listing budget
      (``min(threshold_pct`` of context — default 5% — ``, listing_max_tokens)``):
      bridge + skills-style listing (name + short description per tool),
      degrading to a names-only listing when the full form is over budget.
    - Tier 2 — per-tool listing over budget even names-only (e.g.
      Cloudflare's flat API surface, ~3,300 tools whose names alone are
      ~32K tokens): bare bridge + a one-line-per-server summary (server
      name + tool count) so the model still knows WHICH domains are
      reachable; individual tools are discoverable only via ``tool_search``.
* The catalog is stateless across turns and tools-array assemblies. It is
  rebuilt from the current tool-defs list every time. This is the lesson
  from OpenClaw's cron regression (openclaw/openclaw#84141): a session-keyed
  catalog that drifts out of sync with the live tool registry produces
  silent tool dropouts.
* Bridge tools route through ``model_tools.handle_function_call`` exactly
  like a direct call, so guardrails, plugin pre/post hooks, approval flows,
  and tool-result truncation all fire identically.
* Display and trajectory unwrap is implemented here so the user (CLI activity
  feed, gateway, saved trajectories) always sees the underlying tool, not
  the bridge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.registry import tool_error

logger = logging.getLogger("tools.tool_search")


# Bridge tool names. These names are reserved and may not collide with a
# user/plugin/MCP tool — registration of any tool with these names is
# rejected by the registry's existing override-protection logic.
TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"
SKILL_SEARCH_NAME = "skill_search"

BRIDGE_TOOL_NAMES = frozenset({
    TOOL_SEARCH_NAME,
    TOOL_DESCRIBE_NAME,
    TOOL_CALL_NAME,
    SKILL_SEARCH_NAME,
})

MODEL_VISIBLE_BRIDGE_NAMES = frozenset({
    TOOL_SEARCH_NAME,
    SKILL_SEARCH_NAME,
})

DEFAULT_MAX_HYDRATED_TOOLS = 16

# When estimating tokens from char count without a real tokenizer, this is
# the cheap rule of thumb that's stable across providers. Roughly 4 chars
# per token for English+JSON. Underestimating leads to false negatives
# (tool search not activated when it should); overestimating leads to false
# positives (activated when not needed). 4.0 errs slightly toward
# underestimating, which is the safer default.
CHARS_PER_TOKEN = 4.0


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""

    enabled: str  # "auto" | "on" | "off"
    # Listing budget as a percentage of the model's context window. Under
    # tiered disclosure this no longer gates *activation* (any deferrable
    # tool activates the bridge) — it bounds how much context the embedded
    # catalog listing may consume before disclosure degrades:
    # full listing -> names-only -> bare bridge (tier 2).
    threshold_pct: float  # 0..100
    search_default_limit: int
    max_search_limit: int
    # Catalog listing ("skills-style" progressive disclosure): when active,
    # a grouped name + short-description manifest of every deferred tool is
    # embedded in the tool_search bridge description, so capabilities stay
    # DISCOVERABLE (like the skills listing in the system prompt) while full
    # schemas stay deferred.  "auto" = include when it fits the listing
    # budget (falls back to names-only, then to none = bare bridge);
    # "on" = same rendering, explicit intent; "off" = always bare bridge.
    listing: str = "auto"  # "auto" | "on" | "off"
    # Absolute cap on the embedded listing, regardless of context size.
    # Effective budget = min(listing_max_tokens, threshold_pct% of context).
    listing_max_tokens: int = 4000

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolSearchConfig":
        """Build a config from a raw dict / bool / None.

        Accepts the legacy bool shape (``tools.tool_search: true``) and the
        dict shape (``tools.tool_search: {enabled: auto, ...}``). Validates
        and clamps every numeric field; unknown values fall back to safe
        defaults rather than raising, so a typo in user config does not
        break the agent.
        """
        if raw is True:
            return cls(enabled="auto", threshold_pct=5.0,
                       search_default_limit=5, max_search_limit=5)
        if raw is False:
            return cls(enabled="off", threshold_pct=5.0,
                       search_default_limit=5, max_search_limit=5)
        if not isinstance(raw, dict):
            return cls(enabled="auto", threshold_pct=5.0,
                       search_default_limit=5, max_search_limit=5)

        enabled_raw = str(raw.get("enabled", "auto")).strip().lower()
        if enabled_raw in ("true", "1", "yes"):
            enabled = "on"
        elif enabled_raw in ("false", "0", "no"):
            enabled = "off"
        elif enabled_raw in ("auto", "on", "off"):
            enabled = enabled_raw
        else:
            enabled = "auto"

        threshold_pct = _safe_float(raw.get("threshold_pct"), 5.0)
        threshold_pct = max(0.0, min(100.0, threshold_pct))

        max_search_limit = max(1, min(50, _safe_int(raw.get("max_search_limit"), 5)))
        search_default_limit = max(1, min(max_search_limit,
                                          _safe_int(raw.get("search_default_limit"), 5)))

        listing_raw = str(raw.get("listing", "auto")).strip().lower()
        if listing_raw in ("true", "1", "yes"):
            listing = "on"
        elif listing_raw in ("false", "0", "no"):
            listing = "off"
        elif listing_raw in ("auto", "on", "off"):
            listing = listing_raw
        else:
            listing = "auto"
        listing_max_tokens = max(200, min(60000, _safe_int(raw.get("listing_max_tokens"), 4000)))

        return cls(
            enabled=enabled,
            threshold_pct=threshold_pct,
            search_default_limit=search_default_limit,
            max_search_limit=max_search_limit,
            listing=listing,
            listing_max_tokens=listing_max_tokens,
        )


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_config() -> ToolSearchConfig:
    """Load tool-search config from the user config file."""
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        if not isinstance(tools_cfg, dict):
            tools_cfg = {}
        return ToolSearchConfig.from_raw(tools_cfg.get("tool_search"))
    except Exception as e:
        logger.debug("Failed to load tool-search config: %s", e)
        return ToolSearchConfig.from_raw(None)


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------


def _core_tool_names() -> frozenset[str]:
    """Return the set of tool names that must NEVER be deferred.

    Imported lazily because ``toolsets`` imports from ``tools.registry``
    and we don't want a hard cycle.
    """
    try:
        from toolsets import _HERMES_CORE_TOOLS
        return frozenset(_HERMES_CORE_TOOLS)
    except Exception:
        return frozenset()


# Session-gated GUI toolsets. Off ``_HERMES_CORE_TOOLS`` so non-GUI clients
# never pay their schema; once a session enables them they stay direct.
_DIRECT_SURFACE_TOOLSETS = frozenset({"desktop_ui", "project"})


def is_deferrable_tool_name(name: str) -> bool:
    """Return True if a tool with this name is *eligible* for deferral.

    A tool is deferrable iff it is registered with an MCP toolset prefix
    OR it is neither in ``_HERMES_CORE_TOOLS`` nor a session-gated GUI
    surface toolset. Core and direct surface tools are never deferred even
    when their toolset is technically plugin-provided (this protects
    against accidental shadowing).
    """
    if name in BRIDGE_TOOL_NAMES:
        return False
    if name in _core_tool_names():
        return False
    # Check registry toolset for MCP prefix.
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return False
        if entry.toolset.startswith("mcp-"):
            return True
        if entry.toolset in _DIRECT_SURFACE_TOOLSETS:
            return False
        # Non-MCP, non-core → plugin tool, eligible.
        return True
    except Exception:
        return False


def classify_tools(
    tool_defs: List[Dict[str, Any]],
    *,
    progressive: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a tool-defs list into (visible, deferrable).

    ``visible`` retains every tool that must stay in the model-facing array:
    every core tool, every session-gated GUI surface tool, plus any tool we
    can't classify. ``deferrable`` is the candidate set for catalog entry.
    """
    visible: List[Dict[str, Any]] = []
    deferrable: List[Dict[str, Any]] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if name in BRIDGE_TOOL_NAMES:
            # Should never happen — bridge tools are added after classification —
            # but be defensive.
            continue
        if progressive:
            deferrable.append(td)
        elif is_deferrable_tool_name(name):
            deferrable.append(td)
        else:
            visible.append(td)
    return visible, deferrable


# ---------------------------------------------------------------------------
# Token estimation and threshold gate
# ---------------------------------------------------------------------------


def estimate_tokens_from_schemas(tool_defs: Iterable[Dict[str, Any]]) -> int:
    """Estimate the token cost of a tool-defs list via the chars/4 rule.

    Cheap and stable across providers. The number doesn't need to be exact —
    it gates the activate/skip decision, and a typical 200K context with a
    10% threshold means the decision flips around 20K tokens of schema.
    Order-of-magnitude precision is fine.
    """
    total_chars = 0
    for td in tool_defs:
        try:
            total_chars += len(json.dumps(td, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            total_chars += len(str(td))
    return int(math.ceil(total_chars / CHARS_PER_TOKEN))


def compact_tool_schema(
    tool_def: Dict[str, Any],
    *,
    target_bytes: int = 3072,
    hard_limit_bytes: int = 4096,
) -> Dict[str, Any]:
    """Remove tutorial prose while preserving executable JSON constraints."""
    compact = copy.deepcopy(tool_def)

    def _walk(node: Any, *, root_description: bool = False) -> None:
        if isinstance(node, dict):
            node.pop("examples", None)
            node.pop("$comment", None)
            desc = node.get("description")
            if isinstance(desc, str):
                cap = 600 if root_description else 240
                if len(desc) > cap:
                    node["description"] = desc[: cap - 1].rstrip() + "…"
            for key, value in list(node.items()):
                _walk(value, root_description=(key == "function"))
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(compact)

    def _size() -> int:
        return len(
            json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode()
        )

    if _size() <= target_bytes:
        return compact

    descriptions: List[Tuple[Dict[str, Any], str]] = []

    def _collect(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("description"), str):
                descriptions.append((node, node["description"]))
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for value in node:
                _collect(value)

    _collect(compact)
    for node, description in sorted(
        descriptions, key=lambda item: len(item[1]), reverse=True
    ):
        if _size() <= target_bytes:
            break
        node["description"] = description[:119].rstrip() + "…"
    for node, _ in descriptions:
        if _size() <= hard_limit_bytes:
            break
        node.pop("description", None)

    compact_name = str((compact.get("function") or {}).get("name") or "")
    if _size() > hard_limit_bytes and not compact_name.startswith("mcp__"):
        raise ValueError(
            f"tool schema exceeds hard limit after compaction: {_size()} bytes"
        )
    return compact


def should_activate(
    config: ToolSearchConfig,
    deferrable_tokens: int,
    context_length: Optional[int],
) -> bool:
    """Decide whether tool search should activate for the current assembly.

    ``"off"`` skips unconditionally. ``"on"`` and ``"auto"`` activate whenever
    at least one deferrable tool exists (there's no point swapping a no-op).

    Tiered-disclosure semantics (July 2026): the presence of ANY MCP/plugin
    tool activates the bridge — schemas always defer. What the threshold now
    controls is the *listing budget* (see :func:`listing_token_budget`), not
    activation. ``context_length`` is retained in the signature for
    backward compatibility with existing callers.
    """
    if config.enabled == "off":
        return False
    if deferrable_tokens <= 0:
        return False
    return True


def listing_token_budget(
    config: ToolSearchConfig,
    context_length: Optional[int],
) -> int:
    """Effective token budget for the embedded catalog listing.

    ``min(listing_max_tokens, threshold_pct% of context)``. Without a known
    context size, the percentage leg falls back to a fixed 10K cutoff
    (5% of a typical 200K window).
    """
    if context_length and context_length > 0:
        pct_leg = int(context_length * (config.threshold_pct / 100.0))
    else:
        pct_leg = 10_000
    return max(0, min(config.listing_max_tokens, pct_leg))


# ---------------------------------------------------------------------------
# Catalog + BM25 retrieval
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    """One deferrable tool, in a form the bridge tools can search and serve."""

    name: str
    description: str
    schema: Dict[str, Any]  # The full {"type":"function", "function": {...}} entry.
    source: str  # "mcp" | "plugin" | "other"
    source_name: str  # Toolset name, e.g. "mcp-github" or "kanban"
    schema_hash: str = ""

    # Pre-tokenized fields for BM25.
    _tokens: List[str] = field(default_factory=list)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_QUERY_STOPWORDS = frozenset({
    "a", "an", "and", "for", "in", "of", "on", "or", "the", "to", "tool",
    "tools", "use", "using", "want", "with",
})


def _cjk_tokens(run: str) -> List[str]:
    """Return exact, bigram, and character tokens for a CJK run."""
    if not run:
        return []
    tokens = [run]
    if len(run) > 1:
        tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
        tokens.extend(run)
    return tokens


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text):
        lowered = raw.lower()
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered):
            tokens.extend(_cjk_tokens(lowered))
        elif lowered not in _LATIN_QUERY_STOPWORDS:
            tokens.append(lowered)
    return tokens


def _entry_search_text(td: Dict[str, Any]) -> str:
    """Build the search-text blob for a deferrable tool.

    Includes the tool name (with underscores broken into words so BM25 can
    match against query terms), the description, and argument names and
    descriptions. This mirrors the useful part of Anthropic's Tool Search
    retrieval contract without coupling Hermes to an Anthropic provider.
    """
    fn = td.get("function") or {}
    name = fn.get("name", "")
    desc = fn.get("description", "") or ""
    argument_terms: List[str] = []

    def _collect_arguments(node: Any) -> None:
        if isinstance(node, dict):
            description = node.get("description")
            if isinstance(description, str) and description:
                argument_terms.append(description)
            properties = node.get("properties")
            if isinstance(properties, dict):
                for argument_name, argument_schema in properties.items():
                    argument_terms.append(str(argument_name))
                    _collect_arguments(argument_schema)
            items = node.get("items")
            if items is not None:
                _collect_arguments(items)
            for union_key in ("allOf", "anyOf", "oneOf"):
                variants = node.get(union_key)
                if isinstance(variants, list):
                    for variant in variants:
                        _collect_arguments(variant)
        elif isinstance(node, list):
            for item in node:
                _collect_arguments(item)

    _collect_arguments(fn.get("parameters") or {})
    argument_text = " ".join(argument_terms)
    # Break snake_case and dotted names into words for BM25.
    name_words = name.replace("_", " ").replace(".", " ").replace("-", " ").replace(":", " ")
    # Exact capability names are the most reliable retrieval signal. Repeat
    # them to give dedicated tools priority over generic descriptions that
    # happen to contain more query words.
    return f"{name_words} {name_words} {name_words} {desc} {argument_text}"


def _classify_source(name: str) -> Tuple[str, str]:
    """Return (source_kind, source_name) for a registered tool name."""
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return ("other", "")
        if entry.toolset.startswith("mcp-"):
            return ("mcp", entry.toolset)
        return ("plugin", entry.toolset)
    except Exception:
        return ("other", "")


def build_catalog(tool_defs: List[Dict[str, Any]]) -> List[CatalogEntry]:
    """Build the deferred-tool catalog from a tool-defs list.

    Caller is expected to pass only the deferrable subset (``classify_tools``
    returns it as the second element).
    """
    catalog: List[CatalogEntry] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        desc = fn.get("description", "") or ""
        source, source_name = _classify_source(name)
        entry = CatalogEntry(
            name=name,
            description=desc,
            schema=td,
            source=source,
            source_name=source_name,
            schema_hash=_schema_hash(td),
            _tokens=_tokenize(_entry_search_text(td)),
        )
        catalog.append(entry)
    return catalog


def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                doc_lengths: List[int], avg_dl: float,
                doc_freq: Dict[str, int], n_docs: int,
                k1: float = 1.5, b: float = 0.75) -> float:
    """Standard BM25 score for one query against one document.

    Inlined small implementation rather than adding a dependency. Performance
    is fine — the catalog is bounded by N (tools) typically < 500, and we
    score against the in-memory tokens list.
    """
    if not doc_tokens:
        return 0.0
    score = 0.0
    dl = len(doc_tokens)
    # Pre-count tokens in the doc.
    doc_tf: Dict[str, int] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    for q in query_tokens:
        df = doc_freq.get(q, 0)
        if df == 0:
            continue
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf = doc_tf.get(q, 0)
        if tf == 0:
            continue
        norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
        score += idf * norm
    return score


def search_catalog(catalog: List[CatalogEntry], query: str, limit: int = 5) -> List[CatalogEntry]:
    """Return the top-``limit`` catalog entries for ``query`` by BM25.

    Falls back to a stable name-substring match when BM25 yields no hits
    above zero. That ensures a query like ``"github"`` against a catalog
    where every tool is named ``github_*`` still returns results — BM25
    can underperform when query and document share only one token that
    appears in every document (zero IDF).
    """
    if not catalog or limit <= 0:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Precompute doc statistics.
    doc_lengths = [len(e._tokens) for e in catalog]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq: Dict[str, int] = {}
    for e in catalog:
        seen = set(e._tokens)
        for t in seen:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    n_docs = len(catalog)

    distinct_query_tokens = set(query_tokens)
    require_multiple_terms = len(distinct_query_tokens) >= 3
    normalized_query = " ".join(query_tokens)
    scored: List[Tuple[float, CatalogEntry]] = []
    for entry in catalog:
        matched_terms = distinct_query_tokens.intersection(entry._tokens)
        # A multi-term capability query sharing only one generic word
        # ("send", "search", "image") is not a useful match.
        if require_multiple_terms and len(matched_terms) < 2:
            continue
        s = _bm25_score(query_tokens, entry._tokens, doc_lengths, avg_dl,
                        doc_freq, n_docs)
        name_phrase = " ".join(
            _tokenize(
                entry.name.replace("_", " ").replace(".", " ").replace("-", " ")
            )
        )
        if name_phrase and name_phrase in normalized_query:
            s += 10.0
        if s > 0:
            scored.append((s, entry))

    if not scored:
        # Substring fallback against the original tool name.
        ql = query.lower()
        for entry in catalog:
            if ql in entry.name.lower():
                scored.append((0.1, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


def _short_desc(description: str, max_chars: int = 60) -> str:
    """Return a stable single-line catalog description."""
    text = " ".join(str(description or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _listing_group_label(source_name: str) -> str:
    label = source_name or "other"
    return label[4:] if label.startswith("mcp-") else label


def build_catalog_listing(
    deferrable: List[Dict[str, Any]],
    *,
    max_tokens: int = 4000,
) -> Optional[str]:
    text, _form = build_catalog_listing_with_form(
        deferrable,
        max_tokens=max_tokens,
    )
    return text


def build_catalog_listing_with_form(
    deferrable: List[Dict[str, Any]],
    *,
    max_tokens: int = 4000,
) -> Tuple[Optional[str], str]:
    """Render a deterministic, budgeted deferred-tool catalog."""
    if not deferrable:
        return None, "none"

    groups: Dict[str, List[Tuple[str, str]]] = {}
    for td in deferrable:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        source, source_name = _classify_source(name)
        label = _listing_group_label(
            source_name if source != "other" else "other"
        )
        groups.setdefault(label, []).append(
            (name, _short_desc(fn.get("description", "")))
        )

    if not groups:
        return None, "none"

    def render_group(label: str, mode: str) -> str:
        tools = sorted(groups[label])
        if mode == "summary":
            return (
                f"{label} ({len(tools)} tools — names not listed; "
                f"discover via `{TOOL_SEARCH_NAME}`)"
            )
        lines = [f"{label} tools ({len(tools)}):"]
        if mode == "full":
            for name, desc in tools:
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        else:
            lines.append(", ".join(name for name, _ in tools))
        return "\n".join(lines)

    header = (
        "Deferred tool catalog (find a tool with "
        f"`{TOOL_SEARCH_NAME}` to load its real schema):"
    )

    def assemble(modes: Dict[str, str]) -> str:
        return "\n".join(
            [header]
            + [render_group(label, modes[label]) for label in sorted(groups)]
        )

    def fits(text: str) -> bool:
        return math.ceil(len(text) / CHARS_PER_TOKEN) <= max_tokens

    modes = {label: "full" for label in groups}
    rendered = assemble(modes)
    if fits(rendered):
        return rendered, "full"

    modes = {label: "names" for label in groups}
    rendered = assemble(modes)
    if fits(rendered):
        return rendered, "names"

    by_size = sorted(
        groups,
        key=lambda label: (-len(render_group(label, "names")), label),
    )
    for label in by_size:
        modes[label] = "summary"
        rendered = assemble(modes)
        if fits(rendered):
            form = (
                "groups"
                if all(mode == "summary" for mode in modes.values())
                else "mixed"
            )
            return rendered, form

    return None, "none"


# ---------------------------------------------------------------------------
# Bridge tool schemas
# ---------------------------------------------------------------------------


def _schema_hash(schema: Dict[str, Any]) -> str:
    """Return a stable, short content identity for one sanitized schema."""
    payload = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bridge_tool_schemas(
    deferred_count: Optional[int] = None,
    listing: Optional[str] = None,
    listing_form: str = "",
) -> List[Dict[str, Any]]:
    """Build the model-visible discovery schemas.

    New turns only receive ``tool_search`` and ``skill_search``. Legacy
    ``tool_describe`` and ``tool_call`` dispatch remains below so historical
    sessions can still be replayed safely.

    When ``listing`` is provided (see :func:`build_catalog_listing`), it is
    embedded in the ``tool_search`` description so every deferred capability
    stays *visible* by name — the skills-listing pattern — closing the
    "model doesn't know what it doesn't know" gap while full parameter
    schemas remain deferred. ``listing_form`` selects the framing: per-tool
    forms ("full"/"names") tell the model it may skip the search when it
    sees the exact name; the server-summary form ("groups") tells it which
    DOMAINS are reachable and that search is mandatory for tool discovery.
    """
    # Keep these descriptions byte-for-byte stable across turns. In
    # particular, do not interpolate the number of currently reachable tools:
    # that would invalidate the provider prompt-cache prefix whenever an MCP
    # server or a check_fn changed.
    desc_search = (
        "Search tools available to this session. Matching tool references are "
        "loaded as real tool schemas for the next model step; then call the "
        "loaded tool directly by its real name. Search before concluding that "
        "a deferred capability is unavailable. Searchable domains may include "
        "files/source code, shell/processes, web/research, messaging, schedules, "
        "documents, memory, infrastructure state, and media."
    )
    if listing and listing_form == "groups":
        desc_search += (
            "\n\nThe servers below are connected and their tools ARE available "
            "through this bridge. For any request in these domains, search "
            "here FIRST — do not claim the capability is unavailable and do "
            "not substitute a generic tool (terminal/browser) without "
            "searching.\n\n" + listing
        )
    elif listing:
        desc_search += (
            "\n\nEvery deferred capability is listed below. If a tool name "
            "appears here, do NOT claim it is unavailable — find it with "
            f"`{TOOL_SEARCH_NAME}` so its real schema is loaded."
        )
        if listing_form == "mixed":
            desc_search += (
                " For servers marked 'names not listed', the tools exist "
                f"too — find them with `{TOOL_SEARCH_NAME}` before "
                "concluding anything is missing."
            )
        desc_search += "\n\n" + listing
    desc_skill = (
        "Search up to three task-relevant Skills available to this session. "
        "Returns a short index; load a selected Skill with the real skill_view tool."
    )

    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_NAME,
                "description": desc_search,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords describing the capability you need (e.g. 'create github issue').",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return. Default 5.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": SKILL_SEARCH_NAME,
                "description": desc_skill,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords describing the task or guidance needed.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Public entry point: assemble tool-defs with optional tool search
# ---------------------------------------------------------------------------


@dataclass
class AssemblyResult:
    """Outcome of one assembly. Useful for tests and observability."""

    tool_defs: List[Dict[str, Any]]
    activated: bool
    deferred_count: int = 0
    deferred_tokens: int = 0
    threshold_tokens: int = 0
    # Disclosure tier actually applied:
    #   0 = passthrough (no deferrable tools, or tool_search off)
    #   1 = bridge + catalog listing (full or names-only)
    #   2 = bare bridge — catalog too large for any listing form
    tier: int = 0
    listing_form: str = "none"  # "full" | "names" | "none"


def assemble_progressive_exposure(
    direct_tool_defs: List[Dict[str, Any]],
    deferred_tool_defs: List[Dict[str, Any]],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Build model-visible, reachable, and deferred-only tool snapshots.

    Profile exposure is authoritative: tools from ``direct_tool_defs`` stay
    model-visible, while only ``deferred_tool_defs`` enter the transient
    discovery catalog. If overlapping toolsets resolve to the same concrete
    tool, direct exposure wins.
    """
    direct: List[Dict[str, Any]] = []
    direct_names: set[str] = set()
    for tool_def in direct_tool_defs:
        name = (tool_def.get("function") or {}).get("name", "")
        if not name or name in BRIDGE_TOOL_NAMES or name in direct_names:
            continue
        direct.append(tool_def)
        direct_names.add(name)

    deferred: List[Dict[str, Any]] = []
    deferred_names: set[str] = set()
    for tool_def in deferred_tool_defs:
        name = (tool_def.get("function") or {}).get("name", "")
        if (
            not name
            or name in BRIDGE_TOOL_NAMES
            or name in direct_names
            or name in deferred_names
        ):
            continue
        deferred.append(tool_def)
        deferred_names.add(name)

    reachable = [*direct, *deferred]
    model_visible = [*direct, *bridge_tool_schemas()]
    return model_visible, reachable, deferred


def assemble_tool_defs(
    tool_defs: List[Dict[str, Any]],
    *,
    context_length: Optional[int] = None,
    config: Optional[ToolSearchConfig] = None,
    progressive: bool = False,
) -> AssemblyResult:
    """Return the tool-defs list the model should actually see.

    When tool search is inactive (off, no deferrable tools, or below
    threshold), this is a passthrough. When active, MCP and plugin tools
    are stripped from the visible list and replaced with the three bridge
    tools. Core tools are *never* deferred regardless of config.

    Idempotent: calling with bridge tools already in the input is a no-op
    (they classify as non-core/non-deferrable but their names are reserved,
    so they are filtered out of the deferrable set).
    """
    if config is None:
        config = load_config()

    # Defensive: strip any bridge tools that may already be in the list
    # (e.g. someone called assemble twice).
    incoming = [td for td in tool_defs
                if (td.get("function") or {}).get("name") not in BRIDGE_TOOL_NAMES]

    visible, deferrable = classify_tools(incoming, progressive=progressive)
    if progressive:
        deferred_tokens = estimate_tokens_from_schemas(deferrable)
        return AssemblyResult(
            tool_defs=bridge_tool_schemas(),
            activated=True,
            deferred_count=len(deferrable),
            deferred_tokens=deferred_tokens,
            threshold_tokens=0,
        )
    if not deferrable:
        return AssemblyResult(tool_defs=incoming, activated=False)

    deferrable_tokens = estimate_tokens_from_schemas(deferrable)
    if not should_activate(config, deferrable_tokens, context_length):
        return AssemblyResult(
            tool_defs=incoming,
            activated=False,
            deferred_count=len(deferrable),
            deferred_tokens=deferrable_tokens,
            threshold_tokens=int((context_length or 0) * (config.threshold_pct / 100.0)),
            tier=0,
        )

    listing = None
    listing_form = "none"
    listing_budget = listing_token_budget(config, context_length)
    if config.listing != "off":
        listing, listing_form = build_catalog_listing_with_form(
            deferrable, max_tokens=listing_budget)
    bridge = bridge_tool_schemas(len(deferrable), listing=listing,
                                 listing_form=listing_form)
    result = visible + bridge
    # Tier 1 = per-tool listing for at least part of the catalog (full,
    # names, or mixed). Tier 2 = search-only discovery; the server-level
    # "groups" summary keeps domains visible but individual tools are only
    # reachable via tool_search.
    tier = 1 if listing_form in ("full", "names", "mixed") else 2

    logger.info(
        "tool_search activated (tier %d): %d core/visible tools kept, %d deferred "
        "(~%d tokens), listing %s (budget ~%d tokens)",
        tier, len(visible), len(deferrable), deferrable_tokens,
        listing_form, listing_budget,
    )

    return AssemblyResult(
        tool_defs=result,
        activated=True,
        deferred_count=len(deferrable),
        deferred_tokens=deferrable_tokens,
        threshold_tokens=listing_budget,
        tier=tier,
        listing_form=listing_form,
    )


# ---------------------------------------------------------------------------
# Bridge tool dispatch
# ---------------------------------------------------------------------------


def is_bridge_tool(name: str) -> bool:
    return name in BRIDGE_TOOL_NAMES


def _format_search_hit(entry: CatalogEntry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "source": entry.source,
        "source_name": entry.source_name,
        "schema_hash": entry.schema_hash,
        # Cap description so a chatty MCP server doesn't blow up the result.
        "description": (entry.description or "")[:240],
    }


def _format_tool_reference(entry: CatalogEntry) -> Dict[str, str]:
    return {
        "type": "tool_reference",
        "name": entry.name,
        "schema_hash": entry.schema_hash,
    }


def _available_source_summary(catalog: List[CatalogEntry]) -> List[Dict[str, Any]]:
    """Return a compact, deterministic summary of connected deferred sources.

    Included only when search returns no matches. This gives the model enough
    evidence to retry with a source/action query instead of treating a lexical
    miss as proof that the capability is unavailable, without adding anything
    to the fixed per-turn prompt.
    """
    counts: Dict[str, int] = {}
    for entry in catalog:
        # _listing_group_label already falls back to "other" for empty
        # source names, matching the listing path's grouping.
        label = _listing_group_label(entry.source_name)
        counts[label] = counts.get(label, 0) + 1
    return [
        {"name": name, "tool_count": counts[name]}
        for name in sorted(counts)
    ]


def dispatch_tool_search(args: Dict[str, Any],
                         *,
                         current_tool_defs: List[Dict[str, Any]],
                         config: Optional[ToolSearchConfig] = None,
                         progressive: bool = False) -> str:
    """Execute the ``tool_search`` bridge tool. Returns a JSON string."""
    if config is None:
        config = load_config()
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")

    raw_limit = args.get("limit")
    if raw_limit is None:
        limit = config.search_default_limit
    else:
        limit = max(1, min(config.max_search_limit, _safe_int(raw_limit, config.search_default_limit)))

    _, deferrable = classify_tools(current_tool_defs, progressive=progressive)
    catalog = build_catalog(deferrable)
    hits = search_catalog(catalog, query, limit=limit)
    result: Dict[str, Any] = {
        "query": query,
        "total_available": len(catalog),
        "matches": [_format_search_hit(h) for h in hits],
        "tool_references": [_format_tool_reference(h) for h in hits],
    }
    if not hits and catalog:
        result["available_sources"] = _available_source_summary(catalog)
        result["hint"] = (
            "No lexical match was found, but the sources above are connected "
            "and their tools remain available. Retry tool_search with the "
            "service name plus a concrete action or object before concluding "
            "the capability is unavailable."
        )
    return json.dumps(result, ensure_ascii=False)


def rebuild_hydrated_tool_surface(
    agent: Any,
    names: Optional[Iterable[str]] = None,
    *,
    max_loaded: int = DEFAULT_MAX_HYDRATED_TOOLS,
) -> List[str]:
    """Rebuild the model-visible surface from eager and hydrated real tools.

    Every requested name is re-resolved against the agent's authoritative
    deferred snapshot, so a stale or out-of-scope reference cannot grant a
    capability. The most recently referenced names win when the bounded
    session cache is full.
    """
    if getattr(agent, "_progressive_disclosure", False) is not True:
        return []

    deferred_by_name = {
        (tool_def.get("function") or {}).get("name", ""): tool_def
        for tool_def in (getattr(agent, "deferred_tools", None) or [])
        if (tool_def.get("function") or {}).get("name")
    }
    requested = list(
        names
        if names is not None
        else (getattr(agent, "_hydrated_tool_names", None) or [])
    )
    ordered: List[str] = []
    for name in requested:
        normalized = str(name or "")
        if not normalized or normalized not in deferred_by_name:
            continue
        if normalized in ordered:
            ordered.remove(normalized)
        ordered.append(normalized)
    ordered = ordered[-max(1, int(max_loaded)):]

    direct = list(getattr(agent, "direct_tools", None) or [])
    hydrated = [deferred_by_name[name] for name in ordered]
    model_visible = [*direct, *hydrated, *bridge_tool_schemas()]
    agent._hydrated_tool_names = ordered
    agent.tools = model_visible
    agent.valid_tool_names = {
        (tool_def.get("function") or {}).get("name")
        for tool_def in model_visible
        if (tool_def.get("function") or {}).get("name")
    }
    return ordered


def hydrate_tool_search_result(agent: Any, result: Any) -> Any:
    """Hydrate structured references from one successful tool_search result."""
    if getattr(agent, "_progressive_disclosure", False) is not True:
        return result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError, json.JSONDecodeError):
            return result
    elif isinstance(result, dict):
        payload = copy.deepcopy(result)
    else:
        return result
    if not isinstance(payload, dict) or payload.get("error"):
        return result

    deferred_by_name = {
        (tool_def.get("function") or {}).get("name", ""): tool_def
        for tool_def in (getattr(agent, "deferred_tools", None) or [])
        if (tool_def.get("function") or {}).get("name")
    }
    accepted: List[str] = []
    rejected: List[str] = []
    for reference in payload.get("tool_references") or []:
        if not isinstance(reference, dict) or reference.get("type") != "tool_reference":
            continue
        name = str(reference.get("name") or "")
        tool_def = deferred_by_name.get(name)
        expected_hash = str(reference.get("schema_hash") or "")
        if (
            tool_def is None
            or not expected_hash
            or _schema_hash(tool_def) != expected_hash
        ):
            if name:
                rejected.append(name)
            continue
        accepted.append(name)

    loaded = list(getattr(agent, "_hydrated_tool_names", None) or [])
    for name in accepted:
        if name in loaded:
            loaded.remove(name)
        loaded.append(name)
    active = rebuild_hydrated_tool_surface(agent, loaded)
    active_set = set(active)
    payload["loaded_tools"] = [name for name in accepted if name in active_set]
    evicted = [name for name in loaded if name not in active_set]
    if evicted:
        payload["evicted_tools"] = evicted
    if accepted:
        payload["instruction"] = (
            "The referenced tools are now loaded with their real schemas. "
            "Call them directly by name; do not use tool_describe or tool_call. "
            "Only names in active_hydrated_tools are callable now; earlier "
            "tool_search results may have been evicted. Never substitute a "
            "similarly named tool."
        )
    if rejected:
        payload["rejected_references"] = rejected
    payload["active_hydrated_tools"] = active
    return json.dumps(payload, ensure_ascii=False)


def hydrate_agent_tools_from_messages(
    agent: Any,
    messages: Iterable[Dict[str, Any]],
) -> List[str]:
    """Rebuild session hydration from compact references in conversation history."""
    if getattr(agent, "_progressive_disclosure", False) is not True:
        return []

    search_ids: set[str] = set()
    referenced_names: List[str] = []
    deferred_names = {
        (tool_def.get("function") or {}).get("name", "")
        for tool_def in (getattr(agent, "deferred_tools", None) or [])
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id") or call.get("call_id") or "")
                function = call.get("function") or {}
                name = str(function.get("name") or call.get("name") or "")
                if name == TOOL_SEARCH_NAME and call_id:
                    search_ids.add(call_id)
                elif name in deferred_names:
                    referenced_names.append(name)
        if (
            message.get("role") == "tool"
            and str(message.get("tool_call_id") or "") in search_ids
        ):
            try:
                payload = json.loads(str(message.get("content") or ""))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                for reference in payload.get("tool_references") or []:
                    if isinstance(reference, dict):
                        name = str(reference.get("name") or "")
                        if name:
                            referenced_names.append(name)

    loaded = list(getattr(agent, "_hydrated_tool_names", None) or [])
    loaded.extend(referenced_names)
    return rebuild_hydrated_tool_surface(agent, loaded)


def dispatch_tool_describe(args: Dict[str, Any],
                           *,
                           current_tool_defs: List[Dict[str, Any]],
                           progressive: bool = False) -> str:
    """Execute the ``tool_describe`` bridge tool. Returns a JSON string."""
    name = str(args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name is required"}, ensure_ascii=False)
    if not progressive and not is_deferrable_tool_name(name):
        return json.dumps({
            "error": (
                f"'{name}' is not a deferrable tool. If you see it in the tools list "
                "already, call it directly; otherwise check the spelling against tool_search."
            ),
        }, ensure_ascii=False)
    _, deferrable = classify_tools(current_tool_defs, progressive=progressive)
    for td in deferrable:
        fn = td.get("function") or {}
        if fn.get("name") == name:
            delivered = td if name.startswith("mcp__") else compact_tool_schema(td)
            delivered_fn = delivered.get("function") or {}
            return json.dumps({
                "name": name,
                "description": delivered_fn.get("description", ""),
                "parameters": delivered_fn.get("parameters", {}),
                "schema_hash": _schema_hash(td),
            }, ensure_ascii=False)
    return tool_error(
        f"'{name}' is not currently available. Re-run tool_search to refresh."
    )


def scoped_deferrable_names(
    tool_defs: List[Dict[str, Any]],
    *,
    progressive: bool = False,
) -> frozenset[str]:
    """Return the set of deferrable tool names present in ``tool_defs``.

    ``tool_defs`` is expected to be the *pre-assembly* tool list for the
    current session's toolset scope (i.e. what
    ``get_tool_definitions(skip_tool_search_assembly=True)`` returns for the
    session's enabled/disabled toolsets). The resulting set is the universe of
    tools the session may legitimately reach through ``tool_call``. Used as a
    scoping gate by both the ``model_tools`` bridge dispatch and the
    ``tool_executor`` unwrap so a restricted-toolset session can never invoke
    an out-of-scope tool via the bridge.
    """
    names: set[str] = set()
    for td in tool_defs:
        name = (td.get("function") or {}).get("name", "")
        if name and (progressive or is_deferrable_tool_name(name)):
            names.add(name)
    return frozenset(names)


def validate_deferred_call_args(name: str, args: Dict[str, Any]) -> Optional[str]:
    """Probe-validate ``tool_call`` arguments against the deferred tool schema.

    Only absent schema-required keys are rejected. Calls that cannot be
    validated confidently continue to the underlying tool unchanged.
    """
    try:
        from tools.registry import registry as _registry

        schema = _registry.get_schema(name)
        if not isinstance(schema, dict):
            return None
        fn = schema.get("function") if schema.get("type") == "function" else schema
        if not isinstance(fn, dict):
            return None
        params = fn.get("parameters")
        if not isinstance(params, dict):
            return None
        required = params.get("required")
        if not isinstance(required, list) or not required:
            return None
        missing = [key for key in required if isinstance(key, str) and key not in args]
        if not missing:
            return None
        return json.dumps(
            {
                "error": (
                    f"tool_call to '{name}' is missing required argument(s): "
                    f"{', '.join(missing)}. The tool was NOT invoked."
                ),
                "parameters": params,
                "hint": (
                    "Retry tool_call with 'arguments' matching the parameters "
                    "schema above."
                ),
            },
            ensure_ascii=False,
        )
    except Exception:  # pragma: no cover - validation must never block dispatch
        logger.debug(
            "validate_deferred_call_args failed for %s", name, exc_info=True
        )
        return None


def resolve_underlying_call(
    args: Dict[str, Any],
    *,
    progressive: bool = False,
) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Parse a ``tool_call`` invocation into (underlying_name, args, error_msg).

    Used by:
    * the dispatcher in ``model_tools.handle_function_call``,
    * the display layer (so the activity feed shows the underlying tool),
    * the trajectory recorder.

    On parse error, returns ``(None, {}, error_message)``.
    """
    name = str(args.get("name") or "").strip()
    if not name:
        return None, {}, "tool_call requires a 'name' argument"
    if name in BRIDGE_TOOL_NAMES:
        return None, {}, f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
    raw_args = args.get("arguments")
    if raw_args is None:
        raw_args = {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return None, {}, f"tool_call 'arguments' is not valid JSON: {e}"
    if not isinstance(raw_args, dict):
        return None, {}, "tool_call 'arguments' must be an object"
    if not progressive and not is_deferrable_tool_name(name):
        return None, {}, (
            f"'{name}' is not a deferrable tool. If it appears in the model-facing tools "
            "list already, call it directly instead of via tool_call."
        )
    return name, raw_args, None


def dispatch_skill_search(
    args: Dict[str, Any],
    *,
    available_tools: Optional[set[str]] = None,
    available_toolsets: Optional[set[str]] = None,
) -> str:
    """Search the current profile's Skill metadata without loading bodies."""
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)
    try:
        from tools.skills_tool import search_skills

        matches = search_skills(
            query=query,
            limit=3,
            available_tools=available_tools,
            available_toolsets=available_toolsets,
        )
    except Exception as exc:
        logger.warning("skill_search failed: %s", exc)
        matches = []
    return json.dumps(
        {"query": query, "matches": matches[:3]},
        ensure_ascii=False,
    )


__all__ = [
    "TOOL_SEARCH_NAME",
    "TOOL_DESCRIBE_NAME",
    "TOOL_CALL_NAME",
    "SKILL_SEARCH_NAME",
    "BRIDGE_TOOL_NAMES",
    "MODEL_VISIBLE_BRIDGE_NAMES",
    "DEFAULT_MAX_HYDRATED_TOOLS",
    "ToolSearchConfig",
    "CatalogEntry",
    "AssemblyResult",
    "load_config",
    "is_deferrable_tool_name",
    "classify_tools",
    "estimate_tokens_from_schemas",
    "should_activate",
    "build_catalog",
    "build_catalog_listing",
    "build_catalog_listing_with_form",
    "listing_token_budget",
    "search_catalog",
    "bridge_tool_schemas",
    "assemble_tool_defs",
    "is_bridge_tool",
    "dispatch_tool_search",
    "hydrate_tool_search_result",
    "hydrate_agent_tools_from_messages",
    "rebuild_hydrated_tool_surface",
    "dispatch_tool_describe",
    "dispatch_skill_search",
    "resolve_underlying_call",
    "scoped_deferrable_names",
    "validate_deferred_call_args",
]
