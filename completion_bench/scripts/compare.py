#!/usr/bin/env python3
"""Check mask/acceptance controls between two saved measurement runs."""
import argparse
import json
from pathlib import Path


def load(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    measurements = [r for r in records if r["kind"] == "measurement"]
    if not measurements or any(r["exit_code"] != 0 for r in measurements):
        raise ValueError(f"missing or failed measurements: {path}")
    results = {r["result"]["case"]: r["result"] for r in measurements}
    if len(results) != len(measurements):
        raise ValueError(f"duplicate cases: {path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left, right = load(args.left), load(args.right)
    if left.keys() != right.keys():
        raise ValueError("workload sets differ")
    controls = ("input_bytes", "max_items_in_row", "step_max_items", "mask_checksum", "accepting")
    for name in left:
        for key in controls:
            if left[name][key] != right[name][key]:
                raise ValueError(f"{name}: {key} differs")
        if not left[name]["accepting"]:
            raise ValueError(f"{name}: not accepting")
        print(f"{name}: matching mask checksum and acceptance")
    print(f"Compared {len(left)} workloads; timing is not an equivalence control.")


if __name__ == "__main__":
    main()
