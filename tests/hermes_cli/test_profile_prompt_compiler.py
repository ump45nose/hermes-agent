from pathlib import Path

from hermes_cli.profiles import create_profile
from hermes_cli.prompt_compiler import load_compiled_prompt, verify_compiled_prompt


def test_fresh_profile_compiles_prompt_and_governance_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = create_profile(
        "research",
        no_alias=True,
        no_skills=True,
        prompt_preset="research",
        prompt_model_family="anthropic",
    )
    loaded = load_compiled_prompt(profile)
    assert loaded is not None
    assert loaded[1]["preset"] == "research"
    assert loaded[1]["model_adapter"]["family"] == "anthropic"
    assert verify_compiled_prompt(profile)["ok"]
    config = (profile / "config.yaml").read_text()
    assert "context_files: false" in config
    assert "coding_context: false" in config


def test_clone_copies_compiled_bytes_without_rebuild(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = create_profile(
        "source",
        no_alias=True,
        no_skills=True,
        prompt_preset="ops",
    )
    target = create_profile(
        "target",
        clone_from="source",
        clone_config=True,
        no_alias=True,
    )
    assert (source / "prompt" / "system.md").read_bytes() == (
        target / "prompt" / "system.md"
    ).read_bytes()
    assert (source / "prompt" / "prompt.lock.yaml").read_bytes() == (
        target / "prompt" / "prompt.lock.yaml"
    ).read_bytes()


def test_new_profile_appears_in_live_roster_without_controller_recompile(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    controller = create_profile(
        "lingjun",
        no_alias=True,
        no_skills=True,
        prompt_preset="lingjun",
    )
    compiled_before = (
        controller / "prompt" / "system.md"
    ).read_bytes()

    create_profile(
        "new-specialist",
        no_alias=True,
        no_skills=True,
        description="New live specialist",
        prompt_preset="default",
    )
    # A scratch/import staging directory is not a routable Profile.
    (tmp_path / "profiles" / "source").mkdir()
    from hermes_cli.kanban_decompose import _build_roster

    roster, names = _build_roster()
    assert "new-specialist" in names
    assert "source" not in names
    assert any(
        entry["name"] == "new-specialist"
        and entry["description"] == "New live specialist"
        for entry in roster
    )
    excluded_roster, excluded_names = _build_roster(
        exclude_names={"lingjun"}
    )
    assert "lingjun" not in excluded_names
    assert all(entry["name"] != "lingjun" for entry in excluded_roster)
    assert (controller / "prompt" / "system.md").read_bytes() == compiled_before
