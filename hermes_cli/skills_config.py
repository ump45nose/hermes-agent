"""Strict allowlist configuration for Hermes Agent skills.

``hermes skills`` stores interactive visibility under ``skills.enabled``.
``skills.cron_only`` remains separate and is available only in cron context.
"""

from typing import List, Set

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config, save_config


def _normalize_skill_names(values) -> Set[str]:
    """Normalize a config value into a set of skill names."""
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    try:
        return {str(v).strip() for v in values if str(v).strip()}
    except TypeError:
        return set()


def get_enabled_skills(config: dict) -> Set[str]:
    """Return the strict interactive skill allowlist."""
    skills_cfg = config.get("skills") or {}
    if not isinstance(skills_cfg, dict):
        return set()
    return _normalize_skill_names(skills_cfg.get("enabled"))


def get_cron_only_skills(config: dict) -> Set[str]:
    """Return skills available to cron jobs but hidden interactively."""
    skills_cfg = config.get("skills") or {}
    if not isinstance(skills_cfg, dict):
        return set()
    return _normalize_skill_names(skills_cfg.get("cron_only"))


def save_enabled_skills(config: dict, enabled: Set[str]) -> None:
    """Persist the strict interactive allowlist."""
    config.setdefault("skills", {})
    config["skills"]["enabled"] = sorted(enabled)
    save_config(config)


def _list_all_skills() -> List[dict]:
    """Return all installed skills regardless of allowlist state."""
    try:
        from tools.skills_tool import _find_all_skills

        return _find_all_skills(include_inactive=True)
    except Exception:
        return []


def _get_categories(skills: List[dict]) -> List[str]:
    """Return sorted unique category names (None -> 'uncategorized')."""
    return sorted({s["category"] or "uncategorized" for s in skills})


def _toggle_by_category(skills: List[dict], enabled: Set[str]) -> Set[str]:
    """Select whole categories into the allowlist."""
    from hermes_cli.curses_ui import curses_checklist

    categories = _get_categories(skills)
    labels = []
    pre_selected = set()
    for index, category in enumerate(categories):
        category_skills = {
            skill["name"]
            for skill in skills
            if (skill["category"] or "uncategorized") == category
        }
        labels.append(f"{category} ({len(category_skills)} skills)")
        if category_skills & enabled:
            pre_selected.add(index)

    chosen = curses_checklist(
        "Categories — select enabled categories",
        labels,
        pre_selected,
        cancel_returns=pre_selected,
    )

    new_enabled = set(enabled)
    for index, category in enumerate(categories):
        category_skills = {
            skill["name"]
            for skill in skills
            if (skill["category"] or "uncategorized") == category
        }
        if index in chosen:
            new_enabled |= category_skills
        else:
            new_enabled -= category_skills
    return new_enabled


def skills_command(args=None):
    """Entry point for ``hermes skills``."""
    from hermes_cli.curses_ui import curses_checklist

    config = load_config()
    skills = _list_all_skills()
    if not skills:
        print(color("  No skills installed.", Colors.DIM))
        return

    print()
    print("  1. Toggle individual skills")
    print("  2. Toggle by category")
    print()
    try:
        mode = input(color("  Select [1]: ", Colors.YELLOW)).strip() or "1"
    except (KeyboardInterrupt, EOFError):
        return

    enabled = get_enabled_skills(config)
    if mode == "2":
        new_enabled = _toggle_by_category(skills, enabled)
    else:
        labels = [
            f"{skill['name']}  ({skill['category'] or 'uncategorized'})  "
            f"—  {skill['description'][:55]}"
            for skill in skills
        ]
        pre_selected = {
            index for index, skill in enumerate(skills)
            if skill["name"] in enabled
        }
        chosen = curses_checklist(
            "Enabled skills",
            labels,
            pre_selected,
            cancel_returns=pre_selected,
        )
        new_enabled = {skills[index]["name"] for index in chosen}

    if new_enabled == enabled:
        print(color("  No changes.", Colors.DIM))
        return

    save_enabled_skills(config, new_enabled)
    print(
        color(
            f"✓ Saved strict allowlist: {len(new_enabled)} enabled.",
            Colors.GREEN,
        )
    )
