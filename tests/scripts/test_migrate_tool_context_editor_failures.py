from pathlib import Path

import yaml

from scripts.migrate_tool_context_editor_failures import migrate


def test_migration_backs_up_and_changes_only_editor_mode(tmp_path: Path):
    root = tmp_path / ".hermes"
    for profile, mode in (("lingjun", "report_only"), ("research", "failures")):
        home = root / "profiles" / profile
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {"default": "example"},
                    "tool_context_editor": {"mode": mode},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    result = migrate(root, ("lingjun", "research"))

    assert result["changes"] == [
        {"profile": "lingjun", "before": "report_only", "after": "failures"},
        {"profile": "research", "before": "failures", "after": "failures"},
    ]
    for profile, previous in (
        ("lingjun", "report_only"),
        ("research", "failures"),
    ):
        current = yaml.safe_load(
            (root / "profiles" / profile / "config.yaml").read_text()
        )
        assert current["model"] == {"default": "example"}
        assert current["tool_context_editor"]["mode"] == "failures"
        backup = (
            Path(result["backup"]) / profile / "config.yaml"
        )
        saved = yaml.safe_load(backup.read_text())
        assert saved["tool_context_editor"]["mode"] == previous
        assert backup.stat().st_mode & 0o777 == 0o600
