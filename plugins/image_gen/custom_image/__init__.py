"""Five globally configured OpenAI Images-compatible providers.

The providers deliberately live behind dedicated ``IMAGE_*`` variables rather
than overloading ``OPENAI_*``.  Their credentials are machine-global while the
selected provider remains profile-local through ``image_gen.provider``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}
_MAX_REFERENCE_IMAGES = 16


def _slot_env_keys(slot: int) -> Tuple[str, str, str]:
    prefix = "IMAGE" if slot == 1 else f"IMAGE_{slot}"
    return (
        f"{prefix}_API_KEY",
        f"{prefix}_BASE_URL",
        f"{prefix}_MODEL",
    )


@contextmanager
def _global_hermes_home() -> Iterator[None]:
    """Resolve config/env reads against the root Hermes home."""
    from hermes_constants import (
        get_default_hermes_root,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(get_default_hermes_root())
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _global_env_value(key: str) -> str:
    from hermes_cli.config import get_env_value_prefer_dotenv

    with _global_hermes_home():
        return str(get_env_value_prefer_dotenv(key) or "").strip()


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load an HTTP(S), data URI, or guarded local image."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        import requests

        response = requests.get(ref, timeout=60)
        response.raise_for_status()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return response.content, name
    if lower.startswith("data:"):
        header, separator, payload = ref.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Only base64 data:image URIs are supported")
        extension = "png"
        if "image/" in header.lower():
            extension = header.lower().split("image/", 1)[1].split(";", 1)[0] or "png"
        return base64.b64decode(payload, validate=True), f"image.{extension}"

    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    path = Path(os.path.expanduser(ref))
    with path.open("rb") as handle:
        return handle.read(), path.name or "image.png"


def _response_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


class CustomImageProvider(ImageGenProvider):
    """One fixed custom endpoint slot."""

    def __init__(self, slot: int):
        if slot not in range(1, 6):
            raise ValueError("custom image slot must be between 1 and 5")
        self.slot = slot
        self.api_key_env, self.base_url_env, self.model_env = _slot_env_keys(slot)

    @property
    def name(self) -> str:
        return f"custom-image-{self.slot}"

    @property
    def display_name(self) -> str:
        return f"Custom Image {self.slot}"

    def _settings(self) -> Tuple[str, str, str]:
        return (
            _global_env_value(self.api_key_env),
            _global_env_value(self.base_url_env).rstrip("/"),
            _global_env_value(self.model_env),
        )

    def is_available(self) -> bool:
        # Keep the selected tool dispatchable even before all three fields are
        # configured. ``generate`` then returns the slot-specific
        # configuration error instead of the generic "tool unavailable"
        # message (and never falls through to another provider).
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        # Model is an editable global env field, not a profile-local catalog
        # choice. Returning no catalog prevents CLI/Dashboard model pickers
        # from duplicating it into ``image_gen.model``.
        return []

    def default_model(self) -> Optional[str]:
        return _global_env_value(self.model_env) or None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": "OpenAI Images-compatible endpoint — text-to-image and image editing",
            "env_vars": [
                {
                    "key": self.api_key_env,
                    "prompt": f"{self.display_name} API key",
                    "scope": "global",
                    "secret": True,
                },
                {
                    "key": self.base_url_env,
                    "prompt": f"{self.display_name} base URL (including /v1 when required)",
                    "scope": "global",
                    "secret": False,
                },
                {
                    "key": self.model_env,
                    "prompt": f"{self.display_name} model",
                    "scope": "global",
                    "secret": False,
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "max_reference_images": _MAX_REFERENCE_IMAGES,
        }

    def _config_error(
        self,
        message: str,
        *,
        prompt: str,
        aspect_ratio: str,
        model: str = "",
    ) -> Dict[str, Any]:
        return error_response(
            error=message,
            error_type="configuration_error",
            provider=self.name,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
        )

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        api_key, base_url, model = self._settings()
        missing = [
            key
            for key, value in (
                (self.api_key_env, api_key),
                (self.base_url_env, base_url),
                (self.model_env, model),
            )
            if not value
        ]
        if missing:
            return self._config_error(
                f"{self.display_name} is missing required setting(s): {', '.join(missing)}",
                prompt=prompt,
                aspect_ratio=aspect,
                model=model,
            )
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return self._config_error(
                f"{self.base_url_env} must be an absolute HTTP(S) URL",
                prompt=prompt,
                aspect_ratio=aspect,
                model=model,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed",
                error_type="missing_dependency",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        sources = sources[:_MAX_REFERENCE_IMAGES]
        is_edit = bool(sources)
        size = _SIZES.get(aspect, _SIZES["square"])

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            if is_edit:
                files = []
                for source in sources:
                    data, filename = _load_image_bytes(source)
                    image = io.BytesIO(data)
                    image.name = filename
                    files.append(image)
                response = client.images.edit(
                    model=model,
                    image=files if len(files) > 1 else files[0],
                    prompt=prompt,
                    size=size,
                    n=1,
                )
            else:
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    n=1,
                )
        except Exception as exc:
            logger.debug("%s request failed", self.name, exc_info=True)
            return error_response(
                error=f"{self.display_name} image request failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = _response_field(response, "data") or []
        if not data:
            return error_response(
                error=f"{self.display_name} returned no image data",
                error_type="empty_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = _response_field(first, "b64_json")
        url = _response_field(first, "url")
        revised_prompt = _response_field(first, "revised_prompt")
        prefix = self.name.replace("-", "_")
        if b64:
            try:
                image_ref = str(save_b64_image(b64, prefix=prefix))
            except Exception as exc:
                return error_response(
                    error=f"Could not save generated image: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif url:
            try:
                image_ref = str(save_url_image(url, prefix=prefix))
            except Exception as exc:
                logger.warning(
                    "%s could not cache image URL %s (%s); returning the URL",
                    self.name,
                    url,
                    exc,
                )
                image_ref = url
        else:
            return error_response(
                error=f"{self.display_name} response contained neither b64_json nor URL",
                error_type="empty_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt
        return success_response(
            image=image_ref,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality="image" if is_edit else "text",
            extra=extra,
        )


def register(ctx) -> None:
    for slot in range(1, 6):
        ctx.register_image_gen_provider(CustomImageProvider(slot))
