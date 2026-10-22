#!/usr/bin/env python3
"""Run each workload in a fresh release process and retain raw timing/RSS data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if sys.platform not in ("darwin", "linux"):
        parser.error("RSS normalization is implemented for macOS and Linux only")
    binary = args.binary.resolve(strict=True)
    if args.iterations <= 0 or args.samples <= 0:
        parser.error("iterations and samples must be positive")
    # Refuse to overwrite evidence from another invocation.
    with args.output.open("x") as output:
        output.write(json.dumps({"kind": "metadata", "platform": platform.platform(),
                                 "machine": platform.machine(), "python": sys.version,
                                 "binary": str(binary),
                                 "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                                 "rss_scope": "whole child process",
                                 "rss_source": "wait4.ru_maxrss", "pid": os.getpid()}) + "\n")
        names = subprocess.check_output([str(binary), "--list"], text=True).splitlines()
        for name in names:
            command = [str(binary), name, str(args.iterations), str(args.samples)]
            # A temporary file avoids stdout-pipe backpressure while wait4 reaps
            # this exact child; no cumulative RUSAGE_CHILDREN attribution.
            import tempfile
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                start = time.perf_counter_ns()
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                wall_ns = time.perf_counter_ns() - start
                stdout.seek(0)
                stderr.seek(0)
                text = stdout.read().decode()
                errors = stderr.read().decode()
            record = {"kind": "measurement", "command": command,
                      "exit_code": child.returncode, "process_wall_ns": wall_ns,
                      "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                      "stderr": errors}
            if child.returncode == 0:
                record["result"] = json.loads(text)
            else:
                record["stdout"] = text
            output.write(json.dumps(record) + "\n")
            output.flush()
            if child.returncode:
                raise SystemExit(f"{name} failed with exit {child.returncode}; see {args.output}")


if __name__ == "__main__":
    main()
