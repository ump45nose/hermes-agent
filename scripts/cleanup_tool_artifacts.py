#!/usr/bin/env python3
"""Delete tool artifacts only after their session has been ended for 7 days."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_home", type=Path)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cutoff = time.time() - max(1, args.days) * 86400
    db_path = args.profile_home / "state.db"
    root = args.profile_home / "artifacts" / "tool-results"
    if not db_path.is_file() or not root.is_dir():
        return 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        ended = {
            str(row[0])
            for row in conn.execute(
                "SELECT id FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?",
                (cutoff,),
            )
        }
    for directory in root.iterdir():
        if directory.is_dir() and directory.name in ended:
            print(directory)
            if args.apply:
                shutil.rmtree(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
