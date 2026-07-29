"""Scope and visibility tests for custom image provider configuration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


class _Provider(ImageGenProvider):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def list_models(self):
        return [{"id": f"{self.name}-model"}]

    def default_model(self):
        return f"{self.name}-model"

    def get_setup_schema(self):
        return {
            "name": self.name,
            "env_vars": [
                {
                    "key": f"{self.name.upper().replace('-', '_')}_API_KEY",
                    "scope": "global",
                    "secret": True,
                }
            ],
        }

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


def _no_subscription(*args, **kwargs):
    return SimpleNamespace(
        account_info=None,
        nous_auth_present=False,
        features={},
    )


def test_global_provider_fields_read_and_write_root_env_from_profile(
    tmp_path, monkeypatch
):
    from hermes_cli import tools_config

    root = tmp_path / "hermes"
    profile = root / "profiles" / "companion"
    profile.mkdir(parents=True)
    (root / ".env").write_text("IMAGE_MODEL=shared-model\n", encoding="utf-8")
    (profile / ".env").write_text("IMAGE_MODEL=profile-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: root,
    )
    field = {
        "key": "IMAGE_MODEL",
        "scope": "global",
        "secret": False,
    }

    assert tools_config.get_provider_env_value(field) == "shared-model"
    tools_config.save_provider_env_value(field, "updated-shared-model")

    assert "IMAGE_MODEL=updated-shared-model" in (root / ".env").read_text()
    assert "IMAGE_MODEL=profile-model" in (profile / ".env").read_text()


def test_disabled_official_and_managed_fal_rows_leave_only_custom_slots(monkeypatch):
    from hermes_cli import tools_config

    monkeypatch.setattr(
        tools_config,
        "get_nous_subscription_features",
        _no_subscription,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered",
        lambda: None,
    )
    official = [
        "openai",
        "openai-codex",
        "fal",
        "xai",
        "openrouter",
        "deepinfra",
        "krea",
    ]
    for name in official + [f"custom-image-{slot}" for slot in range(1, 6)]:
        image_gen_registry.register_provider(_Provider(name))

    config = {
        "plugins": {
            "disabled": [f"image_gen/{name}" for name in official],
        }
    }
    visible = tools_config._visible_providers(
        tools_config.TOOL_CATEGORIES["image_gen"],
        config,
    )

    assert [row.get("image_gen_plugin_name") for row in visible] == [
        f"custom-image-{slot}" for slot in range(1, 6)
    ]
    assert all(
        row.get("managed_nous_feature") != "image_gen"
        for row in visible
    )


def test_disabled_fal_does_not_make_image_toolset_ready_via_nous(monkeypatch):
    from hermes_cli import tools_config

    monkeypatch.setattr(
        tools_config,
        "get_nous_subscription_features",
        lambda *args, **kwargs: SimpleNamespace(
            account_info=None,
            nous_auth_present=True,
            features={
                "image_gen": SimpleNamespace(
                    available=True,
                    managed_by_nous=True,
                )
            },
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered",
        lambda: None,
    )
    image_gen_registry.register_provider(_Provider("custom-image-1"))

    config = {
        "plugins": {"disabled": ["image_gen/fal"]},
        "image_gen": {"provider": "custom-image-1"},
    }
    assert tools_config._toolset_has_keys("image_gen", config) is False
