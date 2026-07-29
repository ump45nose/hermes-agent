"""Creation-time compiler for fixed Hermes profile prompts."""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = 2

MODULES: dict[str, dict[str, Any]] = {
    "core-minimal": {
        "version": 1,
        "description": "Minimal truthfulness, completion, and safety boundary.",
        "text": (
            "如实区分已观察事实、推断和未知项。完成用户要求的实际结果；"
            "受阻时说明阻塞点，不伪造执行、来源或成功状态。"
        ),
    },
    "controller": {
        "version": 5,
        "description": "Own the user goal, route cross-profile work, and judge results.",
        "text": (
            "职责\n"
            "你是当前用户会话的 Controller。你持有用户目标，负责判断处理方式、"
            "拆解和派单、验收各项结果、失败补救，并向用户统一交付；"
            "你不代替专业 Profile 执行其领域工作。\n"
            "\n"
            "自己处理\n"
            "只在任务简单、所需信息已在当前上下文中，并且不需要专业取证、"
            "外部调查或跨 Profile 执行时直接处理，例如解释问题、澄清需求、"
            "整理已有材料或汇总已经取得的结果。\n"
            "\n"
            "走 Kanban\n"
            "需要专业领域判断或执行、外部或多来源取证、跨 Profile 协作时，"
            "先调用 kanban_roster，再调用 kanban_create。执行者明确时指定 assignee；"
            "目标仍模糊、跨多个领域或无法可靠选择执行者时使用 triage=true。"
            "任务卡必须带上原始目标、完成标准、用户指定的来源或路径和交付要求。"
            "派单成功后停止当前执行，由 Harness Park 当前会话；收到终态 receipt 后，"
            "按原目标决定接受、补救、换执行者或询问用户。只有相关任务全部终态且"
            "证据足够，才统一回答用户。\n"
            "\n"
            "边界\n"
            "工具可见不等于 Controller 有权代做专业工作。"
            "你决定路线、拆解、执行者、验收和补救；Harness 负责能力校验、创建与订阅、"
            "Park 与唤醒、Worker 状态和真实 receipt。"
            "不得绕过 Kanban 自行调用专业来源或执行专业操作，不得轮询、模拟或猜测任务状态。"
            "存在未终态的相关任务时，不得提前给出最终答案。\n"
            "\n"
            "复杂链路范例\n"
            "用户要求先调查多来源方案，再在现有环境落地并验证：先用 roster 选择研究"
            "执行者并创建调查任务，成功后 Park；收到研究 receipt 并验收后，再创建运维"
            "落地任务并 Park；若结果不足则创建补救任务，全部终态后再综合交付。"
        ),
        "protocols": ["kanban-controller@1"],
        "required_capabilities": [
            "kanban.roster",
            "kanban.create",
            "kanban.controller_receipt",
        ],
        "allowed_runtime_overlays": ["platform", "cron"],
    },
    "direct-action-minimal": {
        "version": 1,
        "text": "职责内且风险可控的简单动作直接完成；不要把执行意图当成已交付结果。",
    },
    "social-companion": {
        "version": 1,
        "text": (
            "以自然、有主动性的陪伴方式交流，保持人物表达一致。"
            "技术性后台通知应安静，不用系统运维口吻打断对话。"
        ),
    },
    "media-delivery": {
        "version": 1,
        "text": (
            "用户明确或自然地请求自拍、看脸、穿搭、姿势、身体部位或发图时，"
            "必须完成真实媒体生成或复用并交付；角色扮演文字不能替代媒体送达。"
        ),
    },
    "operations": {
        "version": 2,
        "description": "Evidence-led operations diagnosis and authorized delivery.",
        "text": (
            "你负责运维诊断与获授权的变更：以当前运行证据为准，明确权限和变更边界，"
            "交付时分别验证配置、进程、实际链路和用户可见结果。"
        ),
    },
    "change-boundaries": {
        "version": 1,
        "text": (
            "变更应范围明确、可回滚并保留用户现有配置。"
            "动态基础设施事实以 shared-state 和实时观测为准。"
        ),
    },
    "research-parent": {
        "version": 6,
        "description": "Fan out to up to three deep research leaves and synthesize evidence.",
        "text": (
            "你是研究父 Agent：正常研究任务先拆成三个互补且独立的 leaf，"
            "立即用 delegate_task 同步并行分发；父进程不直接搜索、抓取或读取"
            " GitHub 等来源。等待三个 leaf 全部终态后，再综合证据、冲突、失败和未解决项。"
            "分发时不得要求 leaf 自行写文件或指定 Evidence 路径；运行时会自动保存"
            "完整 Evidence bundle。只要求 leaf 返回 claims、source_ids、contradictions、"
            "unexpected_findings、unresolved 五字段 JSON。"
            "正常综合只使用 leaf 返回的结构化 handoff；Evidence bundle 仅在"
            "某个具体结论、冲突或来源缺失时按需钻取，不整包重复读取。"
        ),
        "protocols": ["research-parent@1"],
        "required_capabilities": [
            "research.delegate",
            "research.handoff",
            "github.read",
        ],
        "allowed_runtime_overlays": [
            "platform",
            "kanban-worker",
            "cron",
            "research-leaf",
        ],
    },
    "citation-rigor": {
        "version": 3,
        "text": (
            "关键事实必须能追溯到实际读取的来源；搜索结果只用于发现，必须打开或"
            "抓取具体来源后才能作为证据。严格遵守用户指定的来源适配器和路径，"
            "不得用其他来源替代后声称已满足要求。为证据记录足以重新定位的来源标识、"
            "观察时间和状态；区分来源陈述、交叉验证、样本范围和你的推断，"
            "未读取的对象不得评价。"
        ),
    },
    "resource-curation": {
        "version": 2,
        "description": "Curate and retrieve resources only inside the XP domain.",
        "text": (
            "你只在 XP domain 内整理、检索和复用资源；不向其他 Agent 分发任务，"
            "不跨 Profile 发布或写入其他职责域。"
        ),
    },
    "active-retrieval": {
        "version": 1,
        "text": (
            "遇到当前信息、既有资料或可验证事实时主动使用可用搜索、网页、"
            "本地索引和会话检索，不仅依赖记忆或角色文案。"
        ),
    },
}

for _module in MODULES.values():
    _module.setdefault("description", "")
    _module.setdefault("protocols", [])
    _module.setdefault("required_capabilities", [])
    _module.setdefault("allowed_runtime_overlays", [])

PRESETS: dict[str, tuple[str, ...]] = {
    "default": ("core-minimal",),
    "lingjun": ("core-minimal", "controller", "direct-action-minimal"),
    "companion": ("core-minimal", "social-companion", "media-delivery"),
    "ops": ("core-minimal", "operations", "change-boundaries"),
    "research": ("core-minimal", "research-parent", "citation-rigor"),
    "xp": ("core-minimal", "resource-curation", "active-retrieval"),
}

PRESET_OVERLAYS: dict[str, tuple[str, ...]] = {
    "default": ("platform",),
    "lingjun": ("platform", "cron"),
    "companion": ("platform", "cron"),
    "ops": ("platform", "kanban-worker", "cron"),
    "research": ("platform", "kanban-worker", "cron", "research-leaf"),
    "xp": ("platform", "kanban-worker", "cron"),
}

MODEL_ADAPTERS: dict[str, dict[str, Any]] = {
    "generic": {
        "version": 1,
        "text": "工具或来源可用时先取得证据；不要声称尚未实际完成的动作。",
    },
    "openai": {
        "version": 1,
        "text": "需要工具才能完成时持续调用到可验证终态；不要用计划或伪造输出代替执行。",
    },
    "anthropic": {
        "version": 1,
        "text": "保持 tool use 与 tool result 的对应关系；基于已返回的结果继续。",
    },
    "google": {
        "version": 1,
        "text": "独立检索可并行，写入前先确认目标，完成后验证结果。",
    },
}

PROTOCOL_SURFACES: dict[str, tuple[str, ...]] = {
    "kanban-controller@1": ("cli", "telegram", "weixin", "api_server"),
    "research-parent@1": ("cli", "telegram", "weixin", "api_server", "cron"),
}

CAPABILITY_TOOLSETS: dict[str, str | None] = {
    "kanban.roster": "kanban",
    "kanban.create": "kanban",
    "kanban.controller_receipt": None,
    "research.delegate": "delegation",
    "research.handoff": None,
    "github.read": "github",
}

INTERNAL_PROTOCOL_CAPABILITIES = frozenset(
    {"kanban.controller_receipt", "research.handoff"}
)

READ_ONLY_GITHUB_REMOTE_TOOLS = (
    "get_commit",
    "get_file_contents",
    "get_label",
    "get_latest_release",
    "get_me",
    "get_release_by_tag",
    "get_tag",
    "get_team_members",
    "get_teams",
    "issue_read",
    "list_branches",
    "list_commits",
    "list_issue_fields",
    "list_issue_types",
    "list_issues",
    "list_pull_requests",
    "list_releases",
    "list_repository_collaborators",
    "list_tags",
    "pull_request_read",
    "search_code",
    "search_commits",
    "search_issues",
    "search_pull_requests",
    "search_repositories",
    "search_users",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_model_family(model: str | None) -> str:
    value = str(model or "").lower()
    if any(token in value for token in ("claude", "anthropic")):
        return "anthropic"
    if any(token in value for token in ("gemini", "gemma", "google")):
        return "google"
    if any(token in value for token in ("gpt", "codex", "openai", "grok")):
        return "openai"
    return "generic"


def _module_record(module_id: str) -> dict[str, Any]:
    try:
        spec = MODULES[module_id]
    except KeyError as exc:
        raise ValueError(f"unknown prompt module: {module_id}") from exc
    text = str(spec["text"]).strip()
    return {
        "id": module_id,
        "version": int(spec["version"]),
        "sha256": _sha256_text(text),
        "text": text,
        "protocols": list(spec.get("protocols") or []),
        "required_capabilities": list(spec.get("required_capabilities") or []),
        "allowed_runtime_overlays": list(
            spec.get("allowed_runtime_overlays") or []
        ),
    }


def prompt_module_catalog() -> dict[str, Any]:
    """Return the authoritative registry consumed by CLI and Web Builder."""
    return {
        "schema_version": SCHEMA_VERSION,
        "presets": [
            {
                "id": preset,
                "modules": list(modules),
                "allowed_runtime_overlays": list(PRESET_OVERLAYS[preset]),
            }
            for preset, modules in PRESETS.items()
        ],
        "modules": [
            {
                "id": module_id,
                "version": int(spec["version"]),
                "description": str(spec.get("description") or ""),
                "protocols": list(spec.get("protocols") or []),
                "required_capabilities": list(
                    spec.get("required_capabilities") or []
                ),
                "allowed_runtime_overlays": list(
                    spec.get("allowed_runtime_overlays") or []
                ),
            }
            for module_id, spec in MODULES.items()
        ],
        "model_families": list(MODEL_ADAPTERS),
    }


def extra_modules_from_lock(
    lock: dict[str, Any],
    *,
    preset: str | None = None,
) -> tuple[str, ...]:
    """Preserve lock modules not supplied by the selected preset."""
    preset_id = str(preset or lock.get("preset") or "default")
    base = set(PRESETS.get(preset_id, ()))
    return tuple(
        str(record.get("id"))
        for record in lock.get("modules") or []
        if isinstance(record, dict)
        and record.get("id")
        and str(record["id"]) not in base
    )


def compile_profile_prompt(
    preset: str,
    *,
    extra_modules: Iterable[str] = (),
    model_family: str = "generic",
) -> tuple[str, dict[str, Any]]:
    preset_id = str(preset or "default").lower()
    if preset_id not in PRESETS:
        raise ValueError(
            f"unknown prompt preset {preset!r}; choose from {', '.join(sorted(PRESETS))}"
        )
    family = model_family if model_family in MODEL_ADAPTERS else "generic"
    module_ids = list(PRESETS[preset_id])
    for module_id in extra_modules:
        if module_id not in module_ids:
            module_ids.append(module_id)
    records = [_module_record(module_id) for module_id in module_ids]
    protocols = list(
        dict.fromkeys(
            protocol
            for record in records
            for protocol in record["protocols"]
        )
    )
    required_capabilities = list(
        dict.fromkeys(
            capability
            for record in records
            for capability in record["required_capabilities"]
        )
    )
    unknown_capabilities = [
        capability
        for capability in required_capabilities
        if capability not in CAPABILITY_TOOLSETS
    ]
    if unknown_capabilities:
        raise ValueError(
            "prompt modules require unsupported capabilities: "
            + ", ".join(unknown_capabilities)
        )
    protocol_requirements: dict[str, list[str]] = {}
    for record in records:
        for protocol in record["protocols"]:
            values = protocol_requirements.setdefault(protocol, [])
            for capability in record["required_capabilities"]:
                if capability not in values:
                    values.append(capability)
    adapter = MODEL_ADAPTERS[family]
    adapter_text = str(adapter["text"]).strip()
    sections = [record["text"] for record in records] + [adapter_text]
    compiled = "\n\n".join(section for section in sections if section).strip() + "\n"
    lock = {
        "schema_version": SCHEMA_VERSION,
        "preset": preset_id,
        "modules": [
            {
                key: record[key]
                for key in (
                    "id",
                    "version",
                    "sha256",
                    "protocols",
                    "required_capabilities",
                    "allowed_runtime_overlays",
                )
            }
            for record in records
        ],
        "model_adapter": {
            "family": family,
            "version": int(adapter["version"]),
            "sha256": _sha256_text(adapter_text),
        },
        "compiled_sha256": _sha256_text(compiled),
        "protocols": protocols,
        "required_capabilities": required_capabilities,
        "protocol_requirements": protocol_requirements,
        "protocol_surfaces": {
            protocol: list(PROTOCOL_SURFACES.get(protocol, ()))
            for protocol in protocols
        },
        "runtime_overlays": list(PRESET_OVERLAYS[preset_id]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return compiled, lock


def _exposure_toolsets(raw: Any) -> set[str]:
    if isinstance(raw, dict):
        return {
            str(value)
            for group in ("direct", "deferred")
            for value in (raw.get(group) or [])
            if value
        }
    if isinstance(raw, (list, tuple, set)):
        return {str(value) for value in raw if value}
    return set()


def _planned_capability_config(
    config: dict[str, Any],
    lock: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    import copy

    planned = copy.deepcopy(config)
    platform_map = planned.setdefault("platform_toolsets", {})
    if not isinstance(platform_map, dict):
        raise ValueError("platform_toolsets must be a mapping")
    changes: list[dict[str, str]] = []
    for protocol in lock.get("protocols") or []:
        requirements = (lock.get("protocol_requirements") or {}).get(
            protocol, lock.get("required_capabilities") or []
        )
        for surface in (lock.get("protocol_surfaces") or {}).get(protocol, []):
            raw = platform_map.get(surface)
            if isinstance(raw, dict):
                exposure = raw
                direct = exposure.setdefault("direct", [])
                deferred = exposure.setdefault("deferred", [])
            else:
                direct = []
                deferred = list(raw or []) if isinstance(raw, list) else []
                exposure = {"direct": direct, "deferred": deferred}
                platform_map[surface] = exposure
            if not isinstance(direct, list) or not isinstance(deferred, list):
                raise ValueError(
                    f"platform_toolsets.{surface} direct/deferred must be lists"
                )
            for capability in requirements:
                toolset = CAPABILITY_TOOLSETS.get(str(capability))
                if toolset and toolset not in direct and toolset not in deferred:
                    deferred.append(toolset)
                    changes.append(
                        {
                            "surface": str(surface),
                            "capability": str(capability),
                            "toolset": toolset,
                        }
                    )
    required = set(lock.get("required_capabilities") or [])
    if "github.read" in required:
        mcp_servers = planned.setdefault("mcp_servers", {})
        if not isinstance(mcp_servers, dict):
            raise ValueError("mcp_servers must be a mapping")
        github = mcp_servers.setdefault(
            "github",
            {
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {
                    "Authorization": "Bearer ${MCP_GITHUB_API_KEY}",
                },
                "connect_timeout": 30,
                "enabled": True,
            },
        )
        if not isinstance(github, dict):
            raise ValueError("mcp_servers.github must be a mapping")
        github["enabled"] = True
        github["tools"] = {
            "include": list(READ_ONLY_GITHUB_REMOTE_TOOLS),
        }
        # Local effect policy is authoritative. Do not rely on a remote MCP
        # server self-reporting readOnlyHint correctly.
        github["tool_effects"] = {
            "read_only": list(READ_ONLY_GITHUB_REMOTE_TOOLS),
        }
        changes.append(
            {
                "surface": "mcp_servers.github",
                "capability": "github.read",
                "toolset": "github",
            }
        )
        if "research-leaf" in set(lock.get("runtime_overlays") or []):
            subagent = platform_map.setdefault(
                "subagent",
                {"direct": [], "deferred": []},
            )
            if not isinstance(subagent, dict):
                raise ValueError(
                    "platform_toolsets.subagent must use direct/deferred mapping"
                )
            deferred = subagent.setdefault("deferred", [])
            if "github" not in deferred:
                deferred.append("github")
            delegation = planned.setdefault("delegation", {})
            if not isinstance(delegation, dict):
                raise ValueError("delegation must be a mapping")
            leaf_toolsets = delegation.setdefault(
                "research_leaf_toolsets",
                [
                    "web",
                    "browser",
                    "context7",
                    "smart-search",
                    "tool_artifact",
                ],
            )
            if not isinstance(leaf_toolsets, list):
                raise ValueError(
                    "delegation.research_leaf_toolsets must be a list"
                )
            if "github" not in leaf_toolsets:
                leaf_toolsets.append("github")
    return planned, changes


def provision_prompt_capabilities(
    profile_home: str | Path,
    lock: dict[str, Any],
    *,
    write: bool = True,
) -> dict[str, Any]:
    path = Path(profile_home) / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    planned, changes = _planned_capability_config(config, lock)
    if write and planned != config:
        path.write_text(
            yaml.safe_dump(planned, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
    return {"config": planned, "changes": changes}


def verify_protocol_capabilities(
    profile_home: str | Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    path = Path(profile_home) / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(config, dict):
        config = {}
    platform_map = config.get("platform_toolsets") or {}
    surfaces: dict[str, Any] = {}
    all_ok = True
    for protocol in lock.get("protocols") or []:
        requirements = (lock.get("protocol_requirements") or {}).get(
            protocol, lock.get("required_capabilities") or []
        )
        for surface in (lock.get("protocol_surfaces") or {}).get(protocol, []):
            available = _exposure_toolsets(
                platform_map.get(surface) if isinstance(platform_map, dict) else None
            )
            missing = []
            for capability in requirements:
                toolset = CAPABILITY_TOOLSETS.get(str(capability))
                if capability in INTERNAL_PROTOCOL_CAPABILITIES:
                    continue
                if capability == "github.read":
                    github = (
                        (config.get("mcp_servers") or {}).get("github")
                        if isinstance(config.get("mcp_servers"), dict)
                        else None
                    )
                    include = (
                        ((github.get("tools") or {}).get("include") or [])
                        if isinstance(github, dict)
                        and isinstance(github.get("tools"), dict)
                        else []
                    )
                    include_set = {str(name) for name in include}
                    effect_config = (
                        github.get("tool_effects")
                        if isinstance(github, dict)
                        else None
                    )
                    declared_read_only = {
                        str(name)
                        for name in (
                            (effect_config or {}).get("read_only") or []
                        )
                    } if isinstance(effect_config, dict) else set()
                    if (
                        not isinstance(github, dict)
                        or github.get("enabled") is False
                        or not include_set
                        or not include_set.issubset(
                            set(READ_ONLY_GITHUB_REMOTE_TOOLS)
                        )
                        or not include_set.issubset(declared_read_only)
                    ):
                        missing.append(str(capability))
                        continue
                if toolset and toolset not in available:
                    missing.append(str(capability))
            key = f"{protocol}:{surface}"
            surfaces[key] = {
                "ok": not missing,
                "missing": missing,
                "toolsets": sorted(available),
            }
            all_ok = all_ok and not missing
    return {"ok": all_ok, "surfaces": surfaces}


def validate_runtime_prompt_protocols(
    lock: dict[str, Any],
    *,
    platform: str,
    runtime_role: str,
    reachable_toolsets: Iterable[str],
) -> dict[str, Any]:
    """Validate the active runtime surface before an Agent can start."""
    surface = {
        "api": "api_server",
        "api-server": "api_server",
    }.get(str(platform or "").lower(), str(platform or "").lower())
    overlay = {
        "kanban_worker": "kanban-worker",
        "research_leaf": "research-leaf",
        "cron": "cron",
        "subagent": "subagent",
    }.get(str(runtime_role or "interactive"))
    allowed_overlays = set(lock.get("runtime_overlays") or [])
    missing: list[str] = []
    if "platform" not in allowed_overlays:
        missing.append("runtime_overlay:platform")
    if overlay and overlay not in allowed_overlays:
        missing.append(f"runtime_overlay:{overlay}")

    available = set(reachable_toolsets or [])
    active_protocols: list[str] = []
    for protocol in lock.get("protocols") or []:
        surfaces = set(
            (lock.get("protocol_surfaces") or {}).get(protocol, [])
        )
        if surface not in surfaces:
            continue
        active_protocols.append(str(protocol))
        requirements = (lock.get("protocol_requirements") or {}).get(
            protocol, lock.get("required_capabilities") or []
        )
        for capability in requirements:
            capability = str(capability)
            if capability in INTERNAL_PROTOCOL_CAPABILITIES:
                continue
            toolset = CAPABILITY_TOOLSETS.get(capability)
            if toolset and toolset not in available:
                missing.append(capability)
    return {
        "ok": not missing,
        "platform": surface,
        "runtime_role": runtime_role,
        "protocols": active_protocols,
        "missing": sorted(set(missing)),
        "reachable_toolsets": sorted(available),
    }


def prompt_paths(profile_home: str | Path) -> tuple[Path, Path]:
    directory = Path(profile_home) / "prompt"
    return directory / "system.md", directory / "prompt.lock.yaml"


def write_compiled_prompt(
    profile_home: str | Path,
    *,
    preset: str,
    extra_modules: Iterable[str] = (),
    model_family: str = "generic",
    backup: bool = False,
) -> dict[str, Any]:
    system_path, lock_path = prompt_paths(profile_home)
    compiled, lock = compile_profile_prompt(
        preset, extra_modules=extra_modules, model_family=model_family
    )
    provision_prompt_capabilities(profile_home, lock, write=False)
    system_path.parent.mkdir(parents=True, exist_ok=True)
    system_path.parent.chmod(0o700)
    if backup and (system_path.exists() or lock_path.exists()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = system_path.parent / "backups" / stamp
        backup_dir.mkdir(parents=True, exist_ok=False)
        backup_dir.chmod(0o700)
        for source in (system_path, lock_path):
            if source.exists():
                shutil.copy2(source, backup_dir / source.name)
    system_path.write_text(compiled, encoding="utf-8")
    lock_path.write_text(
        yaml.safe_dump(lock, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.chmod(system_path, 0o600)
    os.chmod(lock_path, 0o600)
    provision_prompt_capabilities(profile_home, lock, write=True)
    return lock


def ensure_profile_governance_config(profile_home: str | Path) -> None:
    """Persist deployment defaults that keep runtime prompts deterministic."""
    path = Path(profile_home) / "config.yaml"
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config root must be a mapping: {path}")
    else:
        payload = {}
    agent = payload.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise ValueError(f"agent config must be a mapping: {path}")
    agent["context_files"] = False
    agent["coding_context"] = False
    agent["environment_probe"] = False
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def load_compiled_prompt(profile_home: str | Path) -> tuple[str, dict[str, Any]] | None:
    system_path, lock_path = prompt_paths(profile_home)
    if not system_path.is_file() or not lock_path.is_file():
        return None
    text = system_path.read_text(encoding="utf-8")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    if not isinstance(lock, dict):
        raise ValueError(f"invalid prompt lock: {lock_path}")
    return text, lock


def verify_compiled_prompt(profile_home: str | Path) -> dict[str, Any]:
    loaded = load_compiled_prompt(profile_home)
    if loaded is None:
        return {"ok": False, "reason": "compiled prompt or lock missing"}
    text, lock = loaded
    actual = _sha256_text(text)
    expected = str(lock.get("compiled_sha256") or "")
    module_checks = []
    for record in lock.get("modules") or []:
        module_id = str(record.get("id") or "")
        try:
            current = _module_record(module_id)
        except ValueError:
            module_checks.append({"id": module_id, "ok": False, "reason": "unknown"})
            continue
        module_checks.append(
            {
                "id": module_id,
                "ok": (
                    int(record.get("version") or 0) == current["version"]
                    and str(record.get("sha256") or "") == current["sha256"]
                ),
                "locked_version": record.get("version"),
                "registry_version": current["version"],
            }
        )
    protocol_check = verify_protocol_capabilities(profile_home, lock)
    hash_ok = bool(expected) and actual == expected
    schema_ok = int(lock.get("schema_version") or 0) == SCHEMA_VERSION
    modules_ok = all(item["ok"] for item in module_checks)
    return {
        "ok": hash_ok and schema_ok and modules_ok and protocol_check["ok"],
        "schema_version": lock.get("schema_version"),
        "schema_ok": schema_ok,
        "hash_ok": hash_ok,
        "expected": expected,
        "actual": actual,
        "preset": lock.get("preset"),
        "model_family": (lock.get("model_adapter") or {}).get("family"),
        "modules": module_checks,
        "protocols": protocol_check,
        "runtime_overlays": lock.get("runtime_overlays") or [],
    }


def render_prompt_diff(
    profile_home: str | Path,
    *,
    preset: str | None = None,
    extra_modules: Iterable[str] = (),
    model_family: str | None = None,
) -> str:
    loaded = load_compiled_prompt(profile_home)
    old_text = loaded[0] if loaded else ""
    old_lock = loaded[1] if loaded else {}
    target_preset = preset or old_lock.get("preset") or "default"
    target_family = (
        model_family
        or (old_lock.get("model_adapter") or {}).get("family")
        or "generic"
    )
    selected_extras = tuple(extra_modules)
    if not selected_extras:
        selected_extras = extra_modules_from_lock(
            old_lock, preset=str(target_preset)
        )
    new_text, new_lock = compile_profile_prompt(
        target_preset,
        extra_modules=selected_extras,
        model_family=target_family,
    )
    prompt_diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile="prompt/system.md",
            tofile="prompt/system.md.new",
        )
    )
    config_path = Path(profile_home) / "config.yaml"
    old_config_text = (
        config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    )
    old_config = yaml.safe_load(old_config_text) or {}
    planned, _changes = _planned_capability_config(old_config, new_lock)
    new_config_text = yaml.safe_dump(
        planned, sort_keys=False, allow_unicode=True
    )
    config_diff = "".join(
        difflib.unified_diff(
            old_config_text.splitlines(keepends=True),
            new_config_text.splitlines(keepends=True),
            fromfile="config.yaml",
            tofile="config.yaml.new",
        )
    )
    return prompt_diff + config_diff
