import json
from pathlib import Path
from types import SimpleNamespace

from tools import profile_selfie_tool as selfie
from tools.tool_search import is_deferrable_tool_name


def _make_profile(tmp_path: Path) -> Path:
    profile_home = tmp_path / "profile"
    reference_dir = profile_home / "images" / "reference"
    reference_dir.mkdir(parents=True)
    (profile_home / "SOUL.md").write_text(
        "identity:\n  appearance:\n    hair: long\n    glasses: false\n",
        encoding="utf-8",
    )
    (reference_dir / "portrait.png").write_bytes(b"reference")
    return profile_home


def test_profile_selfie_is_core_and_not_deferred():
    assert is_deferrable_tool_name("profile_selfie") is False


def test_requirements_are_profile_gated(monkeypatch, tmp_path):
    profile_home = _make_profile(tmp_path)
    monkeypatch.setattr(selfie, "_profile_home", lambda: profile_home)
    monkeypatch.setattr(
        selfie.registry,
        "get_entry",
        lambda name: SimpleNamespace() if name == "image_generate" else None,
    )

    monkeypatch.setattr(selfie, "_active_profile_name", lambda: "companion")
    assert selfie.check_profile_selfie_requirements() is True

    monkeypatch.setattr(selfie, "_active_profile_name", lambda: "lingjun")
    assert selfie.check_profile_selfie_requirements() is True

    monkeypatch.setattr(selfie, "_active_profile_name", lambda: "ops")
    assert selfie.check_profile_selfie_requirements() is False


def test_prompt_uses_profile_appearance_without_hardcoded_identity():
    prompt = selfie._prompt("坐在窗边自拍", "appearance:\nhair: long\nglasses: false")

    assert "坐在窗边自拍" in prompt
    assert "hair: long" in prompt
    assert "short hair" not in prompt
    assert "glasses, body proportions" not in prompt


def test_profile_selfie_uses_current_profile_and_returns_delivery(
    monkeypatch,
    tmp_path,
):
    profile_home = _make_profile(tmp_path)
    provider_image = tmp_path / "provider-result.png"
    provider_image.write_bytes(b"generated-image")
    captured = {}

    def fake_image_generate(args, **kwargs):
        captured.update(args)
        return json.dumps({"success": True, "image": str(provider_image)})

    monkeypatch.setattr(selfie, "_profile_home", lambda: profile_home)
    monkeypatch.setattr(
        selfie.registry,
        "get_entry",
        lambda name: (
            SimpleNamespace(handler=fake_image_generate)
            if name == "image_generate"
            else None
        ),
    )

    result = json.loads(
        selfie.profile_selfie(
            {"request": "刚脱完鞋，拍一下脚", "aspect_ratio": "portrait"}
        )
    )

    assert result["success"] is True
    assert result["references_used"] == 1
    assert result["delivery"].startswith("MEDIA:")
    output = Path(result["image"])
    assert output.is_file()
    assert output.parent == profile_home / "images" / "generated"
    assert captured["image_url"].endswith("portrait.png")
    assert "刚脱完鞋，拍一下脚" in captured["prompt"]
