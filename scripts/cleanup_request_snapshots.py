#!/usr/bin/env python3
"""Remove expired diagnostic request snapshots (default: seven days)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_home", type=Path)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cutoff = time.time() - max(1, args.days) * 86400
    directory = args.profile_home / "observability" / "request-snapshots"
    if not directory.is_dir():
        return 0
    for path in directory.iterdir():
        if path.is_file() and path.stat().st_mtime < cutoff:
            print(path)
            if args.apply:
                path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
