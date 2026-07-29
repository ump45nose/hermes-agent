"""Profile-scoped identity image generation for companion profiles."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.registry import registry


_ALLOWED_PROFILES = {"companion", "lingjun"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_IDENTITY_HINTS = ("selfie", "mirror", "portrait", "face")

PROFILE_SELFIE_SCHEMA = {
    "name": "profile_selfie",
    "description": (
        "Generate and prepare delivery of an identity-consistent image of the "
        "active profile in one call. Use whenever the user asks to see the "
        "profile, including their selfie, appearance, outfit, pose, face, or "
        "body. Do not wait for a second request and do not call image_generate "
        "separately. On success, copy the returned `delivery` value verbatim "
        "into the response so the image is sent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "The requested view or scene, preserving useful wording "
                    "and context from the user's message."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["portrait", "square", "landscape"],
                "default": "portrait",
                "description": (
                    "Use portrait unless the requested composition needs otherwise."
                ),
            },
        },
        "required": ["request"],
    },
}


def _profile_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home().resolve()


def _active_profile_name() -> str:
    from hermes_cli.profiles import get_active_profile_name

    return get_active_profile_name()


def _appearance_from_soul(profile_home: Path) -> str:
    soul = profile_home / "SOUL.md"
    try:
        lines = soul.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    start = None
    base_indent = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("appearance:"):
            start = index
            base_indent = len(line) - len(stripped)
            break
    if start is None:
        return ""

    block = [lines[start].strip()]
    for line in lines[start + 1 :]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped and indent <= base_indent:
            break
        block.append(line.strip())
    return "\n".join(line for line in block if line)


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def _reference_images(profile_home: Path) -> list[str]:
    references = _image_files(profile_home / "images" / "reference")
    references.sort(
        key=lambda path: (
            not any(hint in path.stem.lower() for hint in _IDENTITY_HINTS),
            path.name,
        )
    )

    selected = references[:2]
    generated = _image_files(profile_home / "images" / "generated")
    if generated:
        latest = max(generated, key=lambda path: path.stat().st_mtime)
        if latest not in selected:
            selected.append(latest)
    return [str(path) for path in selected]


def _prompt(request: str, appearance: str) -> str:
    parts = [
        "Generate one natural, photorealistic phone photo of the same person "
        "shown in the reference images.",
        f"Requested view or scene: {request}",
        "Preserve the person's stable identity, facial features, hairstyle, "
        "body proportions, and everyday appearance. Use a candid single-photo "
        "composition with natural anatomy, lighting, and skin texture. No "
        "collage, text, or watermark.",
    ]
    if appearance:
        parts.append(f"Stable profile appearance:\n{appearance}")
    return "\n\n".join(parts)


def _materialize_image(profile_home: Path, source: str) -> Path:
    if source.startswith(("http://", "https://")):
        from agent.image_gen_provider import save_url_image

        source_path = save_url_image(source, prefix="profile_selfie")
    else:
        source_path = Path(source).expanduser().resolve()
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise ValueError("image provider returned no usable local image")

    suffix = source_path.suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        suffix = ".png"
    target_dir = profile_home / "images" / "generated"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = target_dir / f"profile-selfie-{timestamp}-{uuid.uuid4().hex[:8]}{suffix}"
    shutil.copy2(source_path, target)
    if target.stat().st_size <= 0:
        raise ValueError("generated image was empty after saving")
    return target.resolve()


def check_profile_selfie_requirements() -> bool:
    """Expose this core tool only to image-ready companion profiles."""
    if _active_profile_name() not in _ALLOWED_PROFILES:
        return False
    profile_home = _profile_home()
    if not (profile_home / "SOUL.md").is_file():
        return False
    if not _reference_images(profile_home):
        return False
    image_tool = registry.get_entry("image_generate")
    if image_tool is None:
        return False
    image_check = getattr(image_tool, "check_fn", None)
    return image_check is None or bool(image_check())


def profile_selfie(args: dict[str, Any], **kwargs: Any) -> str:
    request = str(args.get("request") or "").strip()
    if not request:
        return json.dumps(
            {"success": False, "error": "request is required"},
            ensure_ascii=False,
        )

    aspect_ratio = str(args.get("aspect_ratio") or "portrait").strip().lower()
    if aspect_ratio not in {"portrait", "square", "landscape"}:
        aspect_ratio = "portrait"

    image_tool = registry.get_entry("image_generate")
    if image_tool is None:
        return json.dumps(
            {"success": False, "error": "image_generate is unavailable"},
            ensure_ascii=False,
        )

    profile_home = _profile_home()
    references = _reference_images(profile_home)
    tool_args: dict[str, Any] = {
        "prompt": _prompt(request, _appearance_from_soul(profile_home)),
        "aspect_ratio": aspect_ratio,
    }
    if references:
        tool_args["image_url"] = references[0]
        if len(references) > 1:
            tool_args["reference_image_urls"] = references[1:]

    try:
        raw = image_tool.handler(tool_args, **kwargs)
        result = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(result, dict):
            raise ValueError("image_generate returned an invalid result")
        if not result.get("success") or not result.get("image"):
            return json.dumps(result, ensure_ascii=False)

        image = _materialize_image(profile_home, str(result["image"]))
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"profile selfie failed: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "image": str(image),
            "delivery": f"MEDIA:{image}",
            "references_used": len(references),
        },
        ensure_ascii=False,
    )


registry.register(
    name="profile_selfie",
    toolset="image_gen",
    schema=PROFILE_SELFIE_SCHEMA,
    handler=profile_selfie,
    check_fn=check_profile_selfie_requirements,
    emoji="📷",
)
