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


SCHEMA_VERSION = 1

MODULES: dict[str, dict[str, Any]] = {
    "core-minimal": {
        "version": 1,
        "text": (
            "如实区分已观察事实、推断和未知项。完成用户要求的实际结果；"
            "受阻时说明阻塞点，不伪造执行、来源或成功状态。"
        ),
    },
    "controller": {
        "version": 1,
        "text": (
            "你负责理解目标、拆解跨职责工作、选择合适执行者并判断最终结果。"
            "只有确需跨 Profile 或并行独立工作时才分发；读取全部终态和证据后再汇总。"
        ),
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
        "version": 1,
        "text": (
            "先诊断并读取当前运行证据，再实施获授权的变更。"
            "分别验证配置、进程、链路和业务可见结果。"
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
        "version": 3,
        "text": (
            "你是研究父 Agent：建立独立假设，最多并行三个研究 leaf，"
            "等待全部终态后综合证据、冲突、失败和未解决项。"
            "分发时不得要求 leaf 自行写文件或指定 Evidence 路径；运行时会自动保存"
            "完整 Evidence bundle。只要求 leaf 返回 claims、source_ids、contradictions、"
            "unexpected_findings、unresolved 五字段 JSON。"
            "正常综合只使用 leaf 返回的结构化 handoff；Evidence bundle 仅在"
            "某个具体结论、冲突或来源缺失时按需钻取，不整包重复读取。"
        ),
    },
    "citation-rigor": {
        "version": 1,
        "text": (
            "关键事实必须能追溯到实际读取的来源；区分来源陈述、交叉验证和你的推断。"
        ),
    },
    "resource-curation": {
        "version": 1,
        "text": (
            "你负责隔离范围内的资源整理、检索和复用，不向其他 Agent 分发任务，"
            "不跨 Profile 发布。"
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

PRESETS: dict[str, tuple[str, ...]] = {
    "default": ("core-minimal",),
    "lingjun": ("core-minimal", "controller", "direct-action-minimal"),
    "companion": ("core-minimal", "social-companion", "media-delivery"),
    "ops": ("core-minimal", "operations", "change-boundaries"),
    "research": ("core-minimal", "research-parent", "citation-rigor"),
    "xp": ("core-minimal", "resource-curation", "active-retrieval"),
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

DEFAULT_OVERLAYS = ("platform", "kanban-worker", "cron", "research-leaf")


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
    }


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
    adapter = MODEL_ADAPTERS[family]
    adapter_text = str(adapter["text"]).strip()
    sections = [record["text"] for record in records] + [adapter_text]
    compiled = "\n\n".join(section for section in sections if section).strip() + "\n"
    lock = {
        "schema_version": SCHEMA_VERSION,
        "preset": preset_id,
        "modules": [
            {key: record[key] for key in ("id", "version", "sha256")}
            for record in records
        ],
        "model_adapter": {
            "family": family,
            "version": int(adapter["version"]),
            "sha256": _sha256_text(adapter_text),
        },
        "compiled_sha256": _sha256_text(compiled),
        "runtime_overlays": list(DEFAULT_OVERLAYS),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return compiled, lock


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
    return {
        "ok": bool(expected) and actual == expected,
        "expected": expected,
        "actual": actual,
        "preset": lock.get("preset"),
        "model_family": (lock.get("model_adapter") or {}).get("family"),
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
    new_text, _ = compile_profile_prompt(
        target_preset,
        extra_modules=extra_modules,
        model_family=target_family,
    )
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile="prompt/system.md",
            tofile="prompt/system.md.new",
        )
    )
