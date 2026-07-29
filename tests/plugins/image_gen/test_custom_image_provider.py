"""Tests for the five configurable OpenAI Images-compatible providers."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.custom_image as custom_image


_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)
_B64_PNG = base64.b64encode(_PNG).decode()


def _response(*, b64=None, url=None):
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=b64, url=url, revised_prompt=None)]
    )


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def provider(monkeypatch):
    values = {
        "IMAGE_API_KEY": "slot-one-key",
        "IMAGE_BASE_URL": "https://images.example.test/v1/",
        "IMAGE_MODEL": "image-model-1",
    }
    monkeypatch.setattr(custom_image, "_global_env_value", values.get)
    return custom_image.CustomImageProvider(1)


def _patched_openai(client):
    module = MagicMock()
    module.OpenAI.return_value = client
    return patch.dict("sys.modules", {"openai": module}), module


class TestMetadata:
    def test_registers_exactly_five_fixed_slots(self):
        ctx = MagicMock()
        custom_image.register(ctx)
        registered = [
            call.args[0].name
            for call in ctx.register_image_gen_provider.call_args_list
        ]
        assert registered == [f"custom-image-{slot}" for slot in range(1, 6)]

    def test_model_is_global_field_not_profile_catalog(self, provider):
        assert provider.list_models() == []
        assert provider.default_model() == "image-model-1"

    @pytest.mark.parametrize("slot", range(1, 6))
    def test_env_schema_is_global_and_only_key_is_secret(self, slot):
        provider = custom_image.CustomImageProvider(slot)
        fields = provider.get_setup_schema()["env_vars"]
        prefix = "IMAGE" if slot == 1 else f"IMAGE_{slot}"
        assert [field["key"] for field in fields] == [
            f"{prefix}_API_KEY",
            f"{prefix}_BASE_URL",
            f"{prefix}_MODEL",
        ]
        assert all(field["scope"] == "global" for field in fields)
        assert [field["secret"] for field in fields] == [True, False, False]


class TestConfiguration:
    def test_missing_configuration_still_dispatches_to_explicit_error(
        self, monkeypatch
    ):
        monkeypatch.setattr(custom_image, "_global_env_value", lambda key: "")
        provider = custom_image.CustomImageProvider(4)
        assert provider.is_available() is True
        result = provider.generate("a cat")
        assert result["error_type"] == "configuration_error"
        assert "Custom Image 4" in result["error"]

    def test_missing_fields_report_slot_specific_configuration_error(self, monkeypatch):
        monkeypatch.setattr(
            custom_image,
            "_global_env_value",
            lambda key: "key" if key == "IMAGE_3_API_KEY" else "",
        )
        result = custom_image.CustomImageProvider(3).generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "configuration_error"
        assert "Custom Image 3" in result["error"]
        assert "IMAGE_3_BASE_URL" in result["error"]
        assert "IMAGE_3_MODEL" in result["error"]

    def test_invalid_base_url_is_rejected_without_api_call(self, monkeypatch):
        values = {
            "IMAGE_API_KEY": "key",
            "IMAGE_BASE_URL": "images.example.test/v1",
            "IMAGE_MODEL": "model",
        }
        monkeypatch.setattr(custom_image, "_global_env_value", values.get)
        result = custom_image.CustomImageProvider(1).generate("a cat")
        assert result["error_type"] == "configuration_error"
        assert "absolute HTTP(S) URL" in result["error"]


class TestGenerate:
    def test_text_to_image_uses_explicit_slot_settings_and_caches_b64(
        self, provider, tmp_path
    ):
        client = MagicMock()
        client.images.generate.return_value = _response(b64=_B64_PNG)
        patched, module = _patched_openai(client)

        with patched:
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["provider"] == "custom-image-1"
        assert result["model"] == "image-model-1"
        assert result["modality"] == "text"
        image = Path(result["image"])
        assert image.parent == tmp_path / "cache" / "images"
        assert image.read_bytes() == _PNG
        module.OpenAI.assert_called_once_with(
            api_key="slot-one-key",
            base_url="https://images.example.test/v1",
        )
        kwargs = client.images.generate.call_args.kwargs
        assert kwargs == {
            "model": "image-model-1",
            "prompt": "a cat",
            "size": "1536x1024",
            "n": 1,
        }
        client.images.edit.assert_not_called()

    def test_image_edit_sends_primary_and_reference_images(self, provider, tmp_path):
        primary = tmp_path / "primary.png"
        reference = tmp_path / "reference.png"
        primary.write_bytes(_PNG)
        reference.write_bytes(_PNG)
        client = MagicMock()
        client.images.edit.return_value = _response(b64=_B64_PNG)
        patched, _ = _patched_openai(client)

        with patched:
            result = provider.generate(
                "make it blue",
                image_url=str(primary),
                reference_image_urls=[str(reference)],
            )

        assert result["success"] is True
        assert result["modality"] == "image"
        kwargs = client.images.edit.call_args.kwargs
        assert kwargs["model"] == "image-model-1"
        assert kwargs["prompt"] == "make it blue"
        assert isinstance(kwargs["image"], list)
        assert [image.name for image in kwargs["image"]] == [
            "primary.png",
            "reference.png",
        ]
        client.images.generate.assert_not_called()

    def test_url_response_is_saved_to_cache(self, provider):
        client = MagicMock()
        client.images.generate.return_value = _response(
            url="https://cdn.example.test/result.png"
        )
        patched, _ = _patched_openai(client)
        cached = Path("/tmp/custom-image-cache.png")

        with patched, patch.object(
            custom_image, "save_url_image", return_value=cached
        ) as save:
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"] == str(cached)
        save.assert_called_once_with(
            "https://cdn.example.test/result.png",
            prefix="custom_image_1",
        )

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("401 unauthorized"),
            TimeoutError("request timed out"),
        ],
    )
    def test_upstream_errors_are_explicit(self, provider, error):
        client = MagicMock()
        client.images.generate.side_effect = error
        patched, _ = _patched_openai(client)

        with patched:
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert str(error) in result["error"]

    @pytest.mark.parametrize(
        "response",
        [
            SimpleNamespace(data=[]),
            {"data": [{}]},
        ],
    )
    def test_malformed_or_empty_response_is_rejected(self, provider, response):
        client = MagicMock()
        client.images.generate.return_value = response
        patched, _ = _patched_openai(client)

        with patched:
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"
