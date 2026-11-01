"""Stdlib regressions: python3 -B -m unittest discover -s completion_bench/scripts.

Cover saved controls and exact-child RSS attribution without a Rust build.
"""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import compare
import measure


def result(name="alpha"):
    return {
        "case": name, "input_bytes": 3, "max_items_in_row": 2000,
        "step_max_items": 50000, "mask_checksum": "0123456789abcdef",
        "accepting": True, "completion_stats_enabled": False,
        "template_elapsed_ns": 1,
        "samples": [{"sample": 0, "iterations": 2, "replay_elapsed_ns": 3,
                     "clone_elapsed_ns": 1, "last_replay_stats": {}}],
    }


def records(names=("alpha", "beta")):
    return [{"kind": "metadata", "expected_cases": list(names),
             "iterations": 2, "samples": 1}] + [
        {"kind": "measurement", "command": ["binary", name, "2", "1"],
         "exit_code": 0, "result": result(name)} for name in names
    ]


class CompareTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in data))
            return compare.load(path)

    def test_valid_saved_controls(self):
        self.assertEqual(set(self.load(records())), {"alpha", "beta"})

    def test_legacy_evidence_remains_readable(self):
        data = records()
        for key in ("expected_cases", "iterations", "samples"):
            data[0].pop(key)
        self.assertEqual(set(self.load(data)), {"alpha", "beta"})

    def test_missing_manifested_case_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load(records()[:-1])

    def test_unexpected_manifested_case_is_rejected(self):
        data = records()
        data[0]["expected_cases"] = ["alpha"]
        with self.assertRaises(ValueError):
            self.load(data)

    def test_invalid_manifest_is_rejected(self):
        for value in (None, [], ["alpha", "alpha"], ["alpha", ""]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                data = records()
                data[0]["expected_cases"] = value
                self.load(data)

    def test_duplicate_measurements_are_rejected(self):
        data = records()
        data.append(copy.deepcopy(data[1]))
        with self.assertRaises(ValueError):
            self.load(data)

    def test_mismatched_command_case_is_rejected(self):
        data = records()
        data[1]["command"][1] = "another_case"
        with self.assertRaises(ValueError):
            self.load(data)

    def test_acceptance_must_be_boolean_true(self):
        for value in (False, 1, "false"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                data = records()
                data[1]["result"]["accepting"] = value
                self.load(data)

    def test_checksum_must_have_the_harness_format(self):
        for value in (123, "broken", "g" * 16):
            with self.subTest(value=value), self.assertRaises(ValueError):
                data = records()
                data[1]["result"]["mask_checksum"] = value
                self.load(data)

    def test_numeric_controls_must_be_integers(self):
        for key, value in (("input_bytes", True), ("max_items_in_row", 0),
                           ("step_max_items", "50000")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                data = records()
                data[1]["result"][key] = value
                self.load(data)

    def test_recorded_counts_must_match_the_command(self):
        data = records()
        data[1]["result"]["samples"][0]["iterations"] = 99
        with self.assertRaises(ValueError):
            self.load(data)

    def test_missing_samples_are_rejected(self):
        data = records()
        data[1]["result"]["samples"] = []
        with self.assertRaises(ValueError):
            self.load(data)

    def test_failed_child_is_rejected(self):
        data = records()
        data[1]["exit_code"] = 1
        with self.assertRaises(ValueError):
            self.load(data)


class MeasureTests(unittest.TestCase):
    def run_measure(self, listing="alpha\n", payload=None, platform="darwin",
                    rss=7, exit_code=0):
        if payload is None:
            payload = json.dumps(result())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.jsonl"
            argv = ["measure.py", sys.executable, "--output", str(output),
                    "--iterations", "2", "--samples", "1"]

            def spawn(command, stdout, stderr):
                stdout.write(payload.encode() if isinstance(payload, str) else payload)
                stderr.write(b"child diagnostic")
                return SimpleNamespace(pid=12345, returncode=None)

            error = None
            with patch.object(sys, "argv", argv), patch.object(sys, "platform", platform), \
                    patch.object(measure.platform, "platform", return_value="test-platform"), \
                    patch.object(subprocess, "check_output", return_value=listing), \
                    patch.object(subprocess, "Popen", side_effect=spawn) as popen, \
                    patch.object(measure.os, "wait4", return_value=(12345, exit_code << 8,
                                 SimpleNamespace(ru_maxrss=rss))) as wait4:
                try:
                    measure.main()
                except (ValueError, SystemExit) as exc:
                    error = exc
            data = [json.loads(line) for line in output.read_text().splitlines()]
            return error, data, popen, wait4

    def test_empty_workload_list_is_rejected(self):
        error, _, popen, _ = self.run_measure(listing="")
        self.assertIsNotNone(error)
        popen.assert_not_called()

    def test_duplicate_workload_list_is_rejected(self):
        error, _, popen, _ = self.run_measure(listing="alpha\nalpha\n")
        self.assertIsNotNone(error)
        popen.assert_not_called()

    def test_declared_workloads_and_counts_are_saved(self):
        error, data, _, _ = self.run_measure()
        self.assertIsNone(error)
        self.assertEqual(data[0]["expected_cases"], ["alpha"])
        self.assertEqual((data[0]["iterations"], data[0]["samples"]), (2, 1))

    def test_wrong_case_and_false_acceptance_fail_immediately(self):
        for key, value in (("case", "wrong"), ("accepting", False)):
            with self.subTest(key=key):
                payload = result()
                payload[key] = value
                error, data, _, _ = self.run_measure(payload=json.dumps(payload))
                self.assertIsNotNone(error)
                self.assertIn("validation_error", data[-1])
                self.assertIn("stdout", data[-1])

    def test_malformed_success_stdout_is_preserved(self):
        error, data, _, _ = self.run_measure(payload="not JSON\n")
        self.assertIsNotNone(error)
        self.assertEqual(data[-1]["stdout"], "not JSON\n")
        self.assertEqual(data[-1]["stderr"], "child diagnostic")
        self.assertIn("validation_error", data[-1])

    def test_invalid_utf8_stdout_still_retains_failure_evidence(self):
        error, data, _, _ = self.run_measure(payload=b"\xff")
        self.assertIsNotNone(error)
        self.assertEqual(data[-1]["stdout"], "\ufffd")
        self.assertIn("validation_error", data[-1])

    def test_failed_child_retains_output(self):
        error, data, _, _ = self.run_measure(payload="failure", exit_code=3)
        self.assertIsNotNone(error)
        self.assertEqual(data[-1]["exit_code"], 3)
        self.assertEqual(data[-1]["stdout"], "failure")

    def test_rss_uses_exact_child_and_platform_units(self):
        for platform, expected in (("darwin", 7), ("linux", 7 * 1024)):
            with self.subTest(platform=platform):
                error, data, _, wait4 = self.run_measure(platform=platform)
                self.assertIsNone(error)
                wait4.assert_called_once_with(12345, 0)
                self.assertEqual(data[-1]["peak_rss_bytes"], expected)

    def test_real_subprocess_is_reaped_and_recorded(self):
        # The launcher is mocked solely to run a Python fixture without creating
        # executable files; Popen, wait4 and temporary streams are real here.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.jsonl"
            argv = ["measure.py", sys.executable, "--output", str(output),
                    "--iterations", "2", "--samples", "1"]
            real_popen = subprocess.Popen

            def spawn(command, **kwargs):
                payload = json.dumps(result(command[1]))
                return real_popen([sys.executable, "-c", "print(" + repr(payload) + ")"],
                                  **kwargs)

            with patch.object(sys, "argv", argv), \
                    patch.object(measure.platform, "platform", return_value="test-platform"), \
                    patch.object(subprocess, "check_output", return_value="alpha\nbeta\n"), \
                    patch.object(subprocess, "Popen", side_effect=spawn):
                measure.main()
            data = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(data), 3)
            for record in data[1:]:
                self.assertEqual(record["exit_code"], 0)
                self.assertGreater(record["peak_rss_bytes"], 0)
                self.assertGreater(record["process_wall_ns"], 0)


if __name__ == "__main__":
    unittest.main()
