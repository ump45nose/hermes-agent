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
