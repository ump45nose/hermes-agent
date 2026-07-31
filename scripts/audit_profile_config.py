#!/usr/bin/env python3
"""Read-only drift audit for cross-profile security and ownership invariants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


SUBJECT_RE = re.compile(r"^user-[0-9a-f]{24}$")


def audit(root: Path) -> dict:
    findings: list[dict[str, str]] = []
    review_exposure: list[tuple[str, str, str]] = []

    for home in sorted((root / "profiles").iterdir()):
        config_path = home / "config.yaml"
        if not home.is_dir() or not config_path.is_file():
            continue
        profile = home.name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        api = ((config.get("platforms") or {}).get("api_server") or {})
        api_enabled = api.get("enabled")
        expected_api = profile == "lingjun"
        if profile in {"lingjun", "companion"} and api_enabled is not expected_api:
            findings.append(
                {
                    "profile": profile,
                    "check": "api_server_owner",
                    "detail": f"expected enabled={expected_api}",
                }
            )

        scopes = (config.get("episode_memory") or {}).get("read_scopes")
        if not isinstance(scopes, list) or not scopes:
            findings.append(
                {
                    "profile": profile,
                    "check": "episode_read_scopes",
                    "detail": "missing or empty",
                }
            )
        else:
            for scope in scopes:
                scope_profile = scope.get("profile") if isinstance(scope, dict) else None
                subjects = scope.get("subjects") if isinstance(scope, dict) else None
                if (
                    not isinstance(scope_profile, str)
                    or not isinstance(subjects, list)
                    or not subjects
                    or any(
                        subject != "self" and not SUBJECT_RE.fullmatch(str(subject))
                        for subject in subjects
                    )
                ):
                    findings.append(
                        {
                            "profile": profile,
                            "check": "episode_read_scopes",
                            "detail": "invalid profile/subject scope",
                        }
                    )
            if profile != "lingjun" and scopes != [
                {"profile": "self", "subjects": ["self"]}
            ]:
                findings.append(
                    {
                        "profile": profile,
                        "check": "episode_self_only",
                        "detail": "non-controller scope is broader than self/self",
                    }
                )

        for surface, exposure in (config.get("platform_toolsets") or {}).items():
            if not isinstance(exposure, dict):
                continue
            for mode in ("direct", "deferred"):
                if "learning_review" in (exposure.get(mode) or []):
                    review_exposure.append((profile, str(surface), mode))

        always_loaded = (
            ((config.get("tools") or {}).get("disclosure") or {}).get(
                "always_loaded"
            )
            or []
        )
        if "learning_review" in always_loaded:
            findings.append(
                {
                    "profile": profile,
                    "check": "learning_review_always_loaded",
                    "detail": "review tool must remain deferred",
                }
            )

    expected_review = [("lingjun", "telegram", "deferred")]
    if review_exposure != expected_review:
        findings.append(
            {
                "profile": "*",
                "check": "learning_review_owner",
                "detail": f"found={review_exposure!r}",
            }
        )
    return {
        "status": "ok" if not findings else "drift",
        "profiles_root": str(root / "profiles"),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/hermes/.hermes"),
    )
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
