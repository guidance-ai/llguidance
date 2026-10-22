#!/usr/bin/env python3
"""Validate saved runs and compare corpus mask/acceptance controls.

New run metadata declares its workload set so partial runs cannot pass. Legacy
records remain readable, with an explicit warning that completeness is unknown.
"""
import argparse
import json
from pathlib import Path
import re
import sys


def validate_names(names):
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("workload names must be nonempty and unique")


def validate_result(result, name, iterations, samples):
    """Check controls and requested work before treating a child as successful."""
    if not isinstance(result, dict) or result.get("case") != name:
        raise ValueError(f"{name}: result case does not match command")
    if result.get("accepting") is not True:
        raise ValueError(f"{name}: accepting must be boolean true")
    checksum = result.get("mask_checksum")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{16}", checksum) is None:
        raise ValueError(f"{name}: invalid mask_checksum")
    for key, minimum in (("input_bytes", 0), ("max_items_in_row", 1), ("step_max_items", 1)):
        value = result.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name}: invalid {key}")
    values = result.get("samples")
    if not isinstance(values, list) or len(values) != samples:
        raise ValueError(f"{name}: sample count does not match command")
    for index, value in enumerate(values):
        if (not isinstance(value, dict) or type(value.get("sample")) is not int
                or value["sample"] != index or type(value.get("iterations")) is not int
                or value["iterations"] != iterations):
            raise ValueError(f"{name}: sample index/iterations do not match command")


def load(path, report_legacy=False):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if (not records or not isinstance(records[0], dict)
            or records[0].get("kind") != "metadata"):
        raise ValueError(f"missing initial metadata: {path}")
    metadata = records[0]
    expected = metadata.get("expected_cases")
    if "expected_cases" in metadata:
        validate_names(expected)
    elif report_legacy:
        print(f"{path}: legacy metadata has no workload manifest; completeness is unverified.",
              file=sys.stderr)
    results = {}
    for record in records[1:]:
        if (not isinstance(record, dict) or record.get("kind") != "measurement"
                or type(record.get("exit_code")) is not int or record["exit_code"] != 0
                or "validation_error" in record):
            raise ValueError(f"invalid or failed measurement: {path}")
        command = record.get("command")
        if (not isinstance(command, list) or len(command) != 4
                or any(not isinstance(arg, str) for arg in command)):
            raise ValueError(f"invalid command: {path}")
        name = command[1]
        validate_names([name])
        iterations, samples = int(command[2]), int(command[3])
        if iterations <= 0 or samples <= 0:
            raise ValueError(f"counts must be positive: {path}")
        for key, count in (("iterations", iterations), ("samples", samples)):
            if key in metadata and (type(metadata[key]) is not int or metadata[key] != count):
                raise ValueError(f"{name}: {key} does not match metadata")
        result = record.get("result")
        validate_result(result, name, iterations, samples)
        if name in results:
            raise ValueError(f"duplicate cases: {path}")
        results[name] = result
    if not results:
        raise ValueError(f"missing measurements: {path}")
    if expected is not None and set(results) != set(expected):
        raise ValueError(f"measurements do not match workload manifest: {path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left, right = load(args.left, report_legacy=True), load(args.right, report_legacy=True)
    if left.keys() != right.keys():
        raise ValueError("workload sets differ")
    controls = ("input_bytes", "max_items_in_row", "step_max_items", "mask_checksum", "accepting")
    for name in left:
        for key in controls:
            if left[name][key] != right[name][key]:
                raise ValueError(f"{name}: {key} differs")
        print(f"{name}: matching mask checksum and acceptance")
    print(f"Compared {len(left)} workloads; timing is not an equivalence control.")


if __name__ == "__main__":
    main()
