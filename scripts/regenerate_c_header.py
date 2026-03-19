#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///

"""Regenerate parser/llguidance.h using cbindgen.

Cross-platform replacement for scripts/cbindgen.sh.
Assumes cbindgen is already installed (cargo install cbindgen).
"""

import argparse
import difflib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Lines matching any of these patterns are stripped from the generated header.
HEADER_STRIP_PATTERNS = [
    re.compile(r"\* # Safety"),
    re.compile(r"\* This function should only be called from C code"),
]

# cbindgen log lines matching this pattern are suppressed.
LOG_SKIP_PATTERN = re.compile(r"Skip .+\(not ")


def run_cbindgen(parser_dir: Path, tmp_dir: Path) -> str:
    """Run cbindgen and return the filtered header text."""
    tmp_header = tmp_dir / "llguidance0.h"

    result = subprocess.run(
        [
            "cbindgen",
            "--config", str(parser_dir / "cbindgen.toml"),
            "--crate", "llguidance",
            "--output", str(tmp_header),
        ],
        cwd=str(parser_dir),
        capture_output=True,
        text=True,
    )

    log_output = result.stdout + result.stderr

    if result.returncode != 0:
        print("Failed to generate llguidance.h", file=sys.stderr)
        print(log_output, file=sys.stderr)
        sys.exit(1)

    # Print log lines, filtering out "Skip …(not …" messages.
    for line in log_output.splitlines():
        if not LOG_SKIP_PATTERN.search(line):
            print(line)

    # Filter the generated header.
    raw_header = tmp_header.read_text(encoding="utf-8")
    filtered_lines = [
        line
        for line in raw_header.splitlines(keepends=True)
        if not any(p.search(line) for p in HEADER_STRIP_PATTERNS)
    ]
    filtered_header = "".join(filtered_lines)

    # Write filtered header to tmp for inspection.
    (tmp_dir / "llguidance.h").write_text(filtered_header, encoding="utf-8")

    return filtered_header


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate parser/llguidance.h using cbindgen.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if the header is out of date (don't overwrite).",
    )
    args = parser.parse_args()

    parser_dir = Path(__file__).resolve().parent.parent / "parser"
    target = parser_dir / "llguidance.h"

    with tempfile.TemporaryDirectory(prefix="llguidance_cbindgen_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        new_header = run_cbindgen(parser_dir, tmp_dir)

        if target.exists():
            old_header = target.read_text(encoding="utf-8")
        else:
            old_header = ""

        if old_header == new_header:
            print("llguidance.h is up to date")
            return

        # Show a unified diff.
        diff = difflib.unified_diff(
            old_header.splitlines(keepends=True),
            new_header.splitlines(keepends=True),
            fromfile="llguidance.h",
            tofile="llguidance.h (generated)",
        )
        sys.stdout.writelines(diff)

        if args.check:
            new_header_path = target.with_suffix(".new.h")
            new_header_path.write_text(new_header, encoding="utf-8")
            print("llguidance.h is out of date")
            print(f"Generated header written to: {new_header_path}",
                  file=sys.stderr)
            sys.exit(1)

        target.write_text(new_header, encoding="utf-8")
        print("Updated llguidance.h")


if __name__ == "__main__":
    main()
