"""Profile-scoped identity image generation for companion profiles."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.registry import registry


_ALLOWED_PROFILES = {"companion", "lingjun"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_IDENTITY_HINTS = ("selfie", "mirror", "portrait", "face")
_REQUEST_REPLACEMENTS = (
    ("写实亲密但不低俗", "写实自然"),
    ("亲昵但不低俗", "自然"),
    ("又被又闻又舔后的", "日常活动后的"),
    ("刚刚这么激烈", "刚运动后"),
    ("老公要检查", "展示细节"),
    ("帮我足一下", "双脚脚掌相对并自然靠拢"),
    ("让我闻闻", "近距离展示"),
    ("奖励式近景", "近景"),
    ("奖励近景", "近景"),
    ("刚做完", "刚活动后"),
    ("验货", "查看细节"),
)
_SCENE_LABELS = (
    (("居家", "家里", "沙发", "home", "sofa"), "居家"),
    (("户外", "街头", "公园", "outdoor", "street", "park"), "户外"),
    (("镜子", "mirror"), "镜前"),
    (("自拍", "selfie"), "自拍"),
    (("全身", "full body"), "全身"),
    (("半身", "upper body"), "半身"),
    (("脸", "面部", "portrait", "face"), "脸部"),
    (("腋下", "腋窝", "armpit"), "腋下"),
    (("手", "hand"), "手部"),
    (("腿", "leg"), "腿部"),
    (("脚", "赤足", "足", "feet", "foot", "barefoot"), "脚部"),
    (("鞋", "shoe"), "鞋子"),
    (("穿搭", "衣服", "outfit"), "穿搭"),
    (("走", "迈步", "walking"), "走路"),
    (("坐", "sitting"), "坐姿"),
    (("站", "standing"), "站姿"),
    (("躺", "lying"), "躺姿"),
    (("特写", "近景", "close-up"), "特写"),
)

PROFILE_SELFIE_SCHEMA = {
    "name": "profile_selfie",
    "description": (
        "为当前 Profile 一次生成并准备交付身份一致的照片。用户要求看当前 "
        "Profile 时使用，包括自拍、长相、穿搭、姿势、脸、身体、全身、腿、脚，"
        "或要求“发你的照片”。不要等待用户再次催促，也不要另行调用 "
        "image_generate。成功后必须把返回的 `delivery` 值原样复制进回复，"
        "确保图片真正发送。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "只描述最终画面中可见的主体、姿势、穿着、构图、环境和光线。"
                    "对敏感或私密上下文做最小的中性化，但不得改变可见画面的原意；"
                    "不要复述关系称谓、行为背景或与画面无关的挑逗性措辞。"
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["portrait", "square", "landscape"],
                "default": "portrait",
                "description": (
                    "默认使用竖图；只有用户要求的构图确有需要时才改用其他比例。"
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


def _reference_image(profile_home: Path) -> str | None:
    references = _image_files(profile_home / "images" / "reference")
    references.sort(
        key=lambda path: (
            not any(hint in path.stem.lower() for hint in _IDENTITY_HINTS),
            path.name,
        )
    )
    if references:
        return str(references[0])
    return None


def _sanitize_request(request: str) -> str:
    sanitized = unicodedata.normalize("NFKC", request).strip()
    for original, replacement in _REQUEST_REPLACEMENTS:
        sanitized = sanitized.replace(original, replacement)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = re.sub(r"([，。；、！？])\1+", r"\1", sanitized)
    return sanitized.strip(" ，。；、")


def _identity_focus(request: str) -> str:
    lowered = request.lower()
    if any(term in lowered for term in ("脚", "赤足", "足", "腿", "feet", "foot", "leg")):
        return "skin tone, overall build, and visible limb proportions"
    if any(term in lowered for term in ("手", "hand")):
        return "skin tone and visible hand proportions"
    if any(term in lowered for term in ("腋下", "腋窝", "armpit", "穿搭", "outfit")):
        return "face when visible, hairstyle, skin tone, and overall build"
    if any(term in lowered for term in ("脸", "自拍", "portrait", "face", "selfie")):
        return "face, hairstyle, and skin tone"
    return "face when visible, hairstyle, skin tone, and overall build"


def _prompt(request: str) -> str:
    return "\n\n".join(
        (
            "Generate one natural, photorealistic phone photo.",
            f"Visible scene: {request}",
            "Use the single reference image only as the identity anchor for the "
            "same clearly adult person. Preserve only the identity details "
            f"relevant to this composition: {_identity_focus(request)}.",
            "Use a candid single-photo composition with natural anatomy, "
            "lighting, and skin texture. No collage, text, or watermark.",
        )
    )


def _scene_label(request: str) -> str:
    lowered = request.lower()
    labels: list[str] = []
    for terms, label in _SCENE_LABELS:
        if any(term in lowered for term in terms) and label not in labels:
            labels.append(label)
        if len(labels) == 4:
            break
    return "-".join(labels) or "场景"


def _materialize_image(profile_home: Path, source: str, request: str) -> Path:
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
    scene = _scene_label(request)
    target = (
        target_dir
        / f"profile-selfie-{scene}-{timestamp}-{uuid.uuid4().hex[:8]}{suffix}"
    )
    shutil.copy2(source_path, target)
    if target.stat().st_size <= 0:
        raise ValueError("generated image was empty after saving")
    return target.resolve()


def check_profile_selfie_requirements() -> bool:
    """Expose this core tool only to image-ready companion profiles."""
    if _active_profile_name() not in _ALLOWED_PROFILES:
        return False
    profile_home = _profile_home()
    if not _reference_image(profile_home):
        return False
    image_tool = registry.get_entry("image_generate")
    if image_tool is None:
        return False
    image_check = getattr(image_tool, "check_fn", None)
    return image_check is None or bool(image_check())


def profile_selfie(args: dict[str, Any], **kwargs: Any) -> str:
    request = _sanitize_request(str(args.get("request") or ""))
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
    reference = _reference_image(profile_home)
    if not reference:
        return json.dumps(
            {
                "success": False,
                "error": "profile_selfie requires one approved reference image",
            },
            ensure_ascii=False,
        )
    prompt = _prompt(request)
    tool_args: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    if reference:
        tool_args["image_url"] = reference

    try:
        raw = image_tool.handler(tool_args, **kwargs)
        result = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(result, dict):
            raise ValueError("image_generate returned an invalid result")
        if not result.get("success") or not result.get("image"):
            return json.dumps(result, ensure_ascii=False)

        image = _materialize_image(
            profile_home,
            str(result["image"]),
            request,
        )
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
            "references_used": 1,
            "prompt": prompt,
            "provider": result.get("provider"),
            "model": result.get("model"),
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
