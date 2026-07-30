"""Creation-time compiler for fixed Hermes profile prompts."""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = 3

STABLE_MODULES: dict[str, dict[str, Any]] = {
    "memory-boundary": {
        "version": 1,
        "description": "Keep durable profile memory separate from shared runtime state.",
        "text": (
            "长期信息按层保存：当前 Profile 私有且长期稳定的用户事实、偏好和可复用经验"
            "写入 Memory；跨 Profile 的当前环境与运行事实写入 shared-state。"
            "任务过程、临时状态和短期结果不写长期记忆。"
        ),
    },
    "active-profile-boundary": {
        "version": 1,
        "description": "Minimal defense-in-depth boundary for profile-private data.",
        "text": (
            "仅操作当前 Hermes Profile；未经用户明确指定，不读取或修改其他 Profile "
            "的私有数据、配置与资产。"
        ),
    },
}

STABLE_PRESETS: dict[str, tuple[str, ...]] = {
    preset: ("memory-boundary", "active-profile-boundary")
    for preset in ("default", "lingjun", "companion", "ops", "research", "xp")
}

RESERVED_STABLE_MODULES: tuple[dict[str, Any], ...] = (
    {"id": "session-search", "version": 0, "status": "reserved"},
    {"id": "skill-governance", "version": 0, "status": "reserved"},
)

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
        "version": 6,
        "description": "Own the user goal, route cross-profile work, and judge results.",
        "text": (
            "职责\n"
            "你是当前用户会话的 Controller。你持有用户目标，负责判断处理方式、"
            "拆解和派单、验收各项结果、失败补救，并向用户统一交付；"
            "最终分析、判断和取舍由你负责。\n"
            "\n"
            "自己处理\n"
            "当前 Profile 已暴露的工具可直接用于读取、检索、分析和完成职责内工作。"
            "工具是否可调用由 Profile allowlist、工具 ACL 和 approval 决定；"
            "不要因为存在 Kanban 就把普通查询、简单检索或本可直接完成的工作派出去。\n"
            "\n"
            "走 Kanban\n"
            "只有工作确实需要另一个 Profile 的隔离身份、专属能力、独立长任务或"
            "跨 Profile 协作时才使用 Kanban。需要派单时可先调用 kanban_roster 了解"
            "当前执行者，再调用 kanban_create；查询 roster 本身不承诺必须派单。"
            "执行者明确时指定 assignee；"
            "目标仍模糊、跨多个领域或无法可靠选择执行者时使用 triage=true。"
            "任务卡必须带上原始目标、完成标准、用户指定的来源或路径和交付要求。"
            "派单成功后停止当前执行，由 Harness Park 当前会话；收到终态 receipt 后，"
            "按原目标决定接受、补救、换执行者或询问用户。只有相关任务全部终态且"
            "证据足够，才统一回答用户。\n"
            "\n"
            "边界\n"
            "你决定路线、拆解、执行者、验收和补救；Harness 只负责创建与订阅、"
            "Park 与唤醒、Worker 租约和真实 receipt，不负责按工具名或任务语义替你选路。"
            "不得轮询、模拟或猜测已派任务的状态。"
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
        "version": 2,
        "text": (
            "职责内且风险可控的简单动作直接完成；不要把执行意图当成已交付结果。"
            "用户要求“保存参考图片”时，先识别当前对话中明确指向的图片，再调用当前"
            " Profile 已开放的保存能力并核对真实结果；不要另行生成无关图片、猜测"
            "保存对象，或在未保存时声称完成。"
        ),
    },
    "social-companion": {
        "version": 2,
        "text": (
            "始终扮演 SOUL.md 中定义的人物，保持身份、关系和表达一致。"
            "以自然、有主动性的陪伴方式交流，保持人物表达一致。"
            "技术性后台通知应安静，不用系统运维口吻打断对话；"
            "不用动作旁白代替自然交流，也不必为了推进对话每轮追问。"
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
        "version": 7,
        "description": "Retrieve and integrate evidence, with optional adaptive delegation.",
        "text": (
            "你负责检索、去重、交叉核验和证据整合。可以直接使用当前 Profile allowlist "
            "暴露的搜索、抓取、GitHub 和本地读取工具。简单或边界清晰的任务直接完成；"
            "只有任务能拆成彼此独立的证据面且并行确实提高覆盖率时，才按需使用"
            " delegate_task 创建少量 leaf，禁止机械地固定创建三个子代理。"
            "给 leaf 的任务只包含必要目标、范围和证据标准；leaf 返回来源、事实、冲突、"
            "缺口和 provenance，运行时可保存完整 Evidence bundle。"
            "你负责把多路材料去重并整合成可追溯的 research dossier。由 Lingjun 或"
            "其他 Controller 委派时，交付整合证据、争议和未知项，不替 Controller 做"
            "最终价值判断；直接面向用户时可基于证据回答。"
            "正常整合优先使用有界 handoff；仅在具体结论、冲突或来源缺失时按需钻取"
            " Evidence bundle，不整包重复读取。"
        ),
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


def _stable_module_record(module_id: str) -> dict[str, Any]:
    try:
        spec = STABLE_MODULES[module_id]
    except KeyError as exc:
        raise ValueError(f"unknown stable prompt module: {module_id}") from exc
    text = str(spec["text"]).strip()
    return {
        "id": module_id,
        "version": int(spec["version"]),
        "sha256": _sha256_text(text),
        "text": text,
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
        "stable_modules": [
            {
                "id": module_id,
                "version": int(spec["version"]),
                "description": str(spec.get("description") or ""),
            }
            for module_id, spec in STABLE_MODULES.items()
        ],
        "reserved_stable_modules": [
            dict(record) for record in RESERVED_STABLE_MODULES
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
        for record in (
            lock.get("profile_modules")
            or lock.get("modules")
            or []
        )
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
        "profile_modules": [
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


def compile_stable_prompt(
    preset: str,
    *,
    system_text: str,
    base_lock: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Compile the locked Stable layer without rewriting authored system.md."""
    preset_id = str(preset or "default").lower()
    if preset_id not in STABLE_PRESETS:
        preset_id = "default"
    records = [
        _stable_module_record(module_id)
        for module_id in STABLE_PRESETS[preset_id]
    ]
    stable_text = (
        "\n\n".join(record["text"] for record in records if record["text"]).strip()
        + "\n"
    )
    lock = dict(base_lock)
    lock["schema_version"] = SCHEMA_VERSION
    lock["preset"] = str(base_lock.get("preset") or preset_id)
    lock.pop("compiled_sha256", None)
    if "modules" in lock and "profile_modules" not in lock:
        lock["profile_modules"] = lock.pop("modules")
    lock["stable"] = {
        "path": "stable.md",
        "modules": [
            {
                key: record[key]
                for key in ("id", "version", "sha256")
            }
            for record in records
        ],
        "compiled_sha256": _sha256_text(stable_text),
    }
    lock["reserved_stable_modules"] = [
        dict(record) for record in RESERVED_STABLE_MODULES
    ]
    lock["authored_system"] = {
        "path": "system.md",
        "observed_sha256": _sha256_text(system_text),
    }
    lock["updated_at"] = datetime.now(timezone.utc).isoformat()
    return stable_text, lock


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
        existed = path.exists()
        path.write_text(
            yaml.safe_dump(planned, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if not existed:
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


def stable_prompt_path(profile_home: str | Path) -> Path:
    return Path(profile_home) / "prompt" / "stable.md"


def _backup_prompt_artifacts(
    prompt_dir: Path,
    sources: Iterable[Path],
) -> Path | None:
    existing = [source for source in sources if source.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = prompt_dir / "backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_dir.chmod(prompt_dir.stat().st_mode & 0o777)
    for source in existing:
        shutil.copy2(source, backup_dir / source.name)
    return backup_dir


def write_compiled_prompt(
    profile_home: str | Path,
    *,
    preset: str,
    extra_modules: Iterable[str] = (),
    model_family: str = "generic",
    backup: bool = False,
) -> dict[str, Any]:
    system_path, lock_path = prompt_paths(profile_home)
    stable_path = stable_prompt_path(profile_home)
    compiled, base_lock = compile_profile_prompt(
        preset, extra_modules=extra_modules, model_family=model_family
    )
    stable, lock = compile_stable_prompt(
        preset,
        system_text=compiled,
        base_lock=base_lock,
    )
    provision_prompt_capabilities(profile_home, lock, write=False)
    prompt_dir_existed = system_path.parent.exists()
    system_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_dir_existed:
        system_path.parent.chmod(0o700)
    if backup:
        _backup_prompt_artifacts(
            system_path.parent,
            (system_path, stable_path, lock_path),
        )
    _atomic_write_text(system_path, compiled)
    _atomic_write_text(stable_path, stable)
    # Publish the lock last. A crash between the artifact replacements leaves a
    # detectable hash mismatch instead of silently accepting mixed versions.
    _atomic_write_text(
        lock_path,
        yaml.safe_dump(lock, sort_keys=False, allow_unicode=True),
    )
    provision_prompt_capabilities(profile_home, lock, write=True)
    return lock


def write_stable_prompt(
    profile_home: str | Path,
    *,
    preset: str,
    extra_modules: Iterable[str] = (),
    model_family: str = "generic",
    backup: bool = False,
) -> dict[str, Any]:
    """Compile Stable and lock metadata while preserving authored system.md."""
    system_path, lock_path = prompt_paths(profile_home)
    stable_path = stable_prompt_path(profile_home)
    if not system_path.is_file():
        raise FileNotFoundError(f"authored prompt missing: {system_path}")
    system_text = system_path.read_text(encoding="utf-8")
    _unused_template, base_lock = compile_profile_prompt(
        preset,
        extra_modules=extra_modules,
        model_family=model_family,
    )
    stable, lock = compile_stable_prompt(
        preset,
        system_text=system_text,
        base_lock=base_lock,
    )
    provision_prompt_capabilities(profile_home, lock, write=False)
    system_path.parent.mkdir(parents=True, exist_ok=True)
    if backup:
        _backup_prompt_artifacts(
            system_path.parent,
            (system_path, stable_path, lock_path),
        )
    _atomic_write_text(stable_path, stable)
    _atomic_write_text(
        lock_path,
        yaml.safe_dump(lock, sort_keys=False, allow_unicode=True),
    )
    provision_prompt_capabilities(profile_home, lock, write=True)
    return lock


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one prompt artifact atomically within its destination dir."""
    target_mode = (
        path.stat().st_mode & 0o777
        if path.exists()
        else 0o600 | (path.parent.stat().st_mode & 0o060)
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, target_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def ensure_profile_governance_config(profile_home: str | Path) -> None:
    """Fill creation-time defaults without overriding Profile policy."""
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
    changed = not path.exists()
    for key, value in {
        "context_files": False,
        "coding_context": False,
        "environment_probe": False,
    }.items():
        if key not in agent:
            agent[key] = value
            changed = True
    editor = payload.setdefault("tool_context_editor", {})
    if not isinstance(editor, dict):
        raise ValueError(f"tool_context_editor config must be a mapping: {path}")
    if "mode" not in editor:
        editor["mode"] = "anthropic"
        changed = True
    trigger = editor.setdefault("trigger", {})
    if not isinstance(trigger, dict):
        raise ValueError(f"tool_context_editor.trigger must be a mapping: {path}")
    if "type" not in trigger:
        trigger["type"] = "input_tokens"
        changed = True
    if "value" not in trigger:
        trigger["value"] = 100_000
        changed = True
    keep = editor.setdefault("keep", {})
    if not isinstance(keep, dict):
        raise ValueError(f"tool_context_editor.keep must be a mapping: {path}")
    if "type" not in keep:
        keep["type"] = "tool_uses"
        changed = True
    if "value" not in keep:
        keep["value"] = 3
        changed = True
    if "clear_tool_inputs" not in editor:
        editor["clear_tool_inputs"] = False
        changed = True
    if not changed:
        return
    existed = path.exists()
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if not existed:
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


def load_runtime_prompt(
    profile_home: str | Path,
) -> tuple[str, str, dict[str, Any]] | None:
    """Load locked Stable plus authored system prompt without hash coupling.

    ``prompt/system.md`` remains the authored runtime source of truth.  Missing
    ``stable.md`` is accepted for legacy Profiles and produces an empty Stable
    layer; Gateway startup never creates or upgrades prompt artifacts.
    """
    system_path, lock_path = prompt_paths(profile_home)
    stable_path = stable_prompt_path(profile_home)
    if not system_path.is_file():
        return None
    system_text = system_path.read_text(encoding="utf-8")
    stable_text = (
        stable_path.read_text(encoding="utf-8")
        if stable_path.is_file()
        else ""
    )
    lock: dict[str, Any] = {}
    if lock_path.is_file():
        loaded = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            lock = loaded
    return stable_text, system_text, lock


def verify_compiled_prompt(profile_home: str | Path) -> dict[str, Any]:
    system_path, lock_path = prompt_paths(profile_home)
    stable_path = stable_prompt_path(profile_home)
    missing = [
        str(path.relative_to(Path(profile_home)))
        for path in (stable_path, system_path, lock_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "ok": False,
            "reason": "prompt artifacts missing",
            "missing": missing,
        }

    stable_text = stable_path.read_text(encoding="utf-8")
    system_text = system_path.read_text(encoding="utf-8")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    if not isinstance(lock, dict):
        return {"ok": False, "reason": "invalid prompt lock"}

    stable_meta = lock.get("stable") or {}
    expected = str(stable_meta.get("compiled_sha256") or "")
    actual = _sha256_text(stable_text)
    module_checks = []
    for record in stable_meta.get("modules") or []:
        module_id = str(record.get("id") or "")
        try:
            current = _stable_module_record(module_id)
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
    reserved_locked = lock.get("reserved_stable_modules") or []
    reserved_ok = reserved_locked == list(RESERVED_STABLE_MODULES)
    stable_path_ok = stable_meta.get("path") == "stable.md"
    protocol_check = verify_protocol_capabilities(profile_home, lock)
    hash_ok = bool(expected) and actual == expected
    schema_ok = int(lock.get("schema_version") or 0) == SCHEMA_VERSION
    expected_module_ids = tuple(
        STABLE_PRESETS.get(str(lock.get("preset") or "default"), ())
    )
    locked_module_ids = tuple(
        str(record.get("id") or "") for record in stable_meta.get("modules") or []
    )
    modules_ok = (
        locked_module_ids == expected_module_ids
        and all(item["ok"] for item in module_checks)
    )

    permission_checks = []
    for path in (stable_path, system_path, lock_path):
        mode = path.stat().st_mode & 0o777
        permission_checks.append(
            {
                "path": str(path.relative_to(Path(profile_home))),
                "mode": f"{mode:04o}",
                "ok": (
                    (mode & 0o600) == 0o600
                    and not bool(mode & 0o007)
                    and not bool(mode & 0o111)
                ),
            }
        )
    prompt_dir = stable_path.parent
    prompt_dir_mode = prompt_dir.stat().st_mode & 0o777
    permission_checks.append(
        {
            "path": str(prompt_dir.relative_to(Path(profile_home))),
            "mode": f"{prompt_dir_mode:04o}",
            "ok": (
                (prompt_dir_mode & 0o700) == 0o700
                and not bool(prompt_dir_mode & 0o007)
            ),
        }
    )
    permissions_ok = all(item["ok"] for item in permission_checks)

    observed_system = str(
        (lock.get("authored_system") or {}).get("observed_sha256") or ""
    )
    system_metadata_ok = (
        (lock.get("authored_system") or {}).get("path") == "system.md"
        and len(observed_system) == 64
    )
    current_system = _sha256_text(system_text)
    preset = str(lock.get("preset") or "default")
    _expected_profile, expected_protocol_lock = compile_profile_prompt(
        preset,
        extra_modules=extra_modules_from_lock(lock, preset=preset),
        model_family=str(
            (lock.get("model_adapter") or {}).get("family") or "generic"
        ),
    )
    protocol_metadata_fields = (
        "protocols",
        "required_capabilities",
        "protocol_requirements",
        "protocol_surfaces",
        "runtime_overlays",
    )
    protocol_metadata_ok = all(
        lock.get(field) == expected_protocol_lock.get(field)
        for field in protocol_metadata_fields
    )
    return {
        "ok": (
            hash_ok
            and schema_ok
            and modules_ok
            and reserved_ok
            and stable_path_ok
            and system_metadata_ok
            and protocol_metadata_ok
            and protocol_check["ok"]
            and permissions_ok
        ),
        "schema_version": lock.get("schema_version"),
        "schema_ok": schema_ok,
        "hash_ok": hash_ok,
        "expected": expected,
        "actual": actual,
        "preset": preset,
        "model_family": (lock.get("model_adapter") or {}).get("family"),
        "stable_modules": module_checks,
        "reserved_stable_modules": {
            "ok": reserved_ok,
            "locked": reserved_locked,
        },
        "system": {
            "metadata_ok": system_metadata_ok,
            "observed_sha256": observed_system,
            "current_sha256": current_system,
            "changed_since_observation": (
                bool(observed_system) and observed_system != current_system
            ),
        },
        "protocol_metadata_ok": protocol_metadata_ok,
        "protocols": protocol_check,
        "runtime_overlays": lock.get("runtime_overlays") or [],
        "permissions": {
            "ok": permissions_ok,
            "artifacts": permission_checks,
        },
    }


def render_prompt_diff(
    profile_home: str | Path,
    *,
    preset: str | None = None,
    extra_modules: Iterable[str] = (),
    model_family: str | None = None,
) -> str:
    loaded = load_compiled_prompt(profile_home)
    system_path, _lock_path = prompt_paths(profile_home)
    system_text = (
        system_path.read_text(encoding="utf-8")
        if system_path.is_file()
        else ""
    )
    old_lock = loaded[1] if loaded else {}
    stable_path = stable_prompt_path(profile_home)
    old_stable_text = (
        stable_path.read_text(encoding="utf-8") if stable_path.is_file() else ""
    )
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
    _profile_text, base_lock = compile_profile_prompt(
        target_preset,
        extra_modules=selected_extras,
        model_family=target_family,
    )
    new_stable_text, new_lock = compile_stable_prompt(
        str(target_preset),
        system_text=system_text,
        base_lock=base_lock,
    )
    prompt_diff = "".join(
        difflib.unified_diff(
            old_stable_text.splitlines(keepends=True),
            new_stable_text.splitlines(keepends=True),
            fromfile="prompt/stable.md",
            tofile="prompt/stable.md.new",
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
