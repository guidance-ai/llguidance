#!/usr/bin/env python3

# This script reads the output.map file and prints the sizes of the sections and modules.
# It works on macOS only.
#
# Usage:
# RUSTFLAGS="-C link-arg=-Wl,-map,output.map" cargo build --release
# python3 rust_size.py output.map
# fx output.json

import json
import os
import re
import subprocess
import sys


def run_command(command, input_data):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=input_data)
    return stdout, stderr


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output.map>")
        sys.exit(1)

    output_map = sys.argv[1]
    if not os.path.exists(output_map):
        print(f"Error: {output_map} does not exist")
        sys.exit(1)

    with open(output_map, "rb") as f:
        file_content = f.read()

    stdout, stderr = run_command(["rustfilt"], file_content)
    if stderr:
        print(f"Error: {stderr.decode('utf-8')}", file=sys.stderr)
        sys.exit(1)

    lib_by_id = {}
    sizes_by_lib = {}
    sizes_by_module = {}
    num_wrong = 0
    min_addr = None
    max_addr = 0
    sections = []

    for line in stdout.decode("utf-8").split("\n"):
        match = re.match(r"\[\s*(\d+)\] .*/([^/]+/[^-\.]+)", line) or re.match(r"\[\s*(\d+)\] ([^/]+)$", line)
        if match:
            lib_id = int(match.group(1))
            lib = match.group(2)
            lib_by_id[lib_id] = lib
            continue
        # 0x1001BEE8C	0x00007F58	__TEXT	__unwind_info
        match = re.match(r"(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+\S+\s+__(\S+)", line, re.IGNORECASE)
        if match:
            start = int(match.group(1), 16)
            end = start + int(match.group(2), 16)
            sections.append((start, end, match.group(3)))
            continue

        # 0x100019778	0x00000180	[ 10] _core::ptr::drop_in_place<llguidance::api::GrammarWithLexer>
        match = re.match(r"(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+\[\s*(\d+)\] (.+)", line, re.IGNORECASE)
        sys_modules = {"core", "std", "alloc", "collections", "panic_unwind", "rustc_demangle"}
        if match:
            addr = int(match.group(1), 16)
            if min_addr is None or addr < min_addr:
                min_addr = addr
            max_addr = max(max_addr, addr)
            size = int(match.group(2), 16)
            lib_id = int(match.group(3))
            name = match.group(4)
            if lib_id not in lib_by_id:
                print(f"Error: lib not found for id {lib_id}")
                continue
            lib = lib_by_id[lib_id]
            section = "unk"
            for start, end, sec in sections:
                if start <= addr < end:
                    section = sec
                    break
            if lib not in sizes_by_lib:
                sizes_by_lib[lib] = {
                    "lib": lib,
                    "total": 0,
                    "sections": {},
                }
            e = sizes_by_lib[lib]
            if section not in e["sections"]:
                e["sections"][section] = 0
            e["sections"][section] += size
            e["total"] += size

            module = None
            for word in re.split(r"[<>&{}\s,\[\]\(\)\.]+", name):
                parts = [w for w in word.split("::") if w]
                if len(parts) > 1:
                    parts[0] = re.sub(r"^_*", "", parts[0])
                    if parts[0] not in sys_modules or module is None or module[0] in sys_modules:
                        module = parts
            if module:
                d = sizes_by_module
                for part in module:
                    if part not in d:
                        d[part] = {"_": 0}
                    d[part]["_"] += size
                    d = d[part]

            continue
        if line.startswith("<<dead>>"):
            continue
        print(line)
        num_wrong += 1
        if num_wrong > 100:
            print("Too many wrong lines. STOP.")
            sys.exit(1)

    total_size = max_addr - min_addr

    entries = sorted(sizes_by_lib.values(), key=lambda x: x["total"], reverse=True)
    for e in entries:
        e["sections"] = dict(sorted(e["sections"].items(), key=lambda x: x[1], reverse=True))

    cutoff_perc = 0.01
    cutoff = total_size / 100 * cutoff_perc

    byfile = {}
    for e in entries:
        lib = e["lib"]
        tsize = e["total"]
        if tsize < cutoff:
            continue
        k = f"{lib:25}: {tsize} ({tsize / total_size * 100:.2f}%)"
        byfile[k] = []
        print(k)
        for section, size in e["sections"].items():
            if size < cutoff:
                continue
            k2 = f"{section:15}: {size} ({size / tsize * 100:.0f}%)"
            byfile[k].append(k2)
            print(f"  {k2}")

    accounted_size = sum(e["total"] for e in sizes_by_lib.values())
    accounted_perc = accounted_size / total_size * 100
    module_accounted_perc = sum(d["_"] for d in sizes_by_module.values()) / total_size * 100
    print(f"Total size: {total_size} ({accounted_perc:.0f}% accounted; {module_accounted_perc:.0f}% by module)")

    bymodule = {}

    def add_rec(name: str, trg: dict, d: dict):
        trg2 = {}
        n = d["_"]
        if n < cutoff:
            return
        trg[f"{name:25}: {n}"] = trg2
        for k, v in sorted(d.items(), key=lambda x: 0 if x[0] == "_" else x[1]["_"], reverse=True):
            if k == "_":
                continue
            add_rec(k, trg2, v)

    for k, v in sorted(sizes_by_module.items(), key=lambda x: x[1]["_"], reverse=True):
        add_rec(k, bymodule, v)

    with open("output.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "byfile": byfile,
                    "bymodule": bymodule,
                    "total_size": total_size,
                    "accounted_size": accounted_size,
                    "accounted_perc": accounted_perc,
                    "module_accounted_perc": module_accounted_perc,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
