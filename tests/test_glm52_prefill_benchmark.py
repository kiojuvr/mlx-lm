import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace

from benchmarks import glm52_prefill_benchmark as benchmark


class TestGlm52PrefillBenchmark(unittest.TestCase):
    def test_extract_checkpoint_summary_reports_lcp_accounting(self):
        summary = benchmark.extract_checkpoint_summary(
            [
                (
                    "prompt checkpoint: prefill summary "
                    "total_prompt_tokens=8192 server_cached_tokens=0 "
                    "disk_cached_tokens=6144 fresh_prompt_tokens=2048 "
                    "fresh_prefill_tokens=2047 prefill_step_size=2048 "
                    "resolution=prefix files_scanned=1 candidates_scanned=1 "
                    "matched_candidates=1 manifest_entries=1 "
                    "manifest_bootstrap=0 lookup_seconds=0.001234"
                )
            ]
        )

        self.assertEqual(summary["checkpoint_total_prompt_tokens"], 8192)
        self.assertEqual(summary["disk_cached_tokens"], 6144)
        self.assertEqual(summary["fresh_prompt_tokens"], 2048)
        self.assertEqual(summary["fresh_prefill_tokens"], 2047)
        self.assertEqual(summary["checkpoint_resolution"], "prefix")
        self.assertEqual(summary["checkpoint_lookup_seconds"], 0.001234)

    def test_format_output_cell_serializes_compound_values(self):
        self.assertEqual(benchmark.format_output_cell({"decode": 156}), '{"decode":156}')
        self.assertEqual(benchmark.format_output_cell(None), "")

    def test_controlled_lcp_rejects_unexpected_checkpoint_length(self):
        args = Namespace(
            lcp_prefix_tokens=4,
            lcp_suffix_tokens=2,
            repeat_prefix_tokens=0,
            repeat_suffix_tokens=0,
            checkpoint_store_prefix_lengths=None,
            checkpoint_frontier_min_tokens=8192,
            no_prompt_checkpoint=False,
            target_tokens=None,
        )
        old_build_prompts = benchmark.build_controlled_lcp_prompts
        old_run_once = benchmark.run_once
        calls = []

        def fake_build_prompts(_tokenizer, prefix_tokens, suffix_tokens):
            requested = prefix_tokens + suffix_tokens
            return list(range(requested)), list(range(requested))

        def fake_run_once(_model, _tokenizer, _prompt, _args, case_name):
            calls.append(case_name)
            disk_cached_tokens = 0 if len(calls) == 1 else 3
            return {
                "case": case_name,
                "disk_cached_tokens": disk_cached_tokens,
                "checkpoint_resolution": "prefix",
            }

        try:
            benchmark.build_controlled_lcp_prompts = fake_build_prompts
            benchmark.run_once = fake_run_once
            with self.assertRaisesRegex(
                RuntimeError,
                "expected disk_cached_tokens=4, got 3",
            ):
                benchmark.run_controlled_lcp(None, None, args)
        finally:
            benchmark.build_controlled_lcp_prompts = old_build_prompts
            benchmark.run_once = old_run_once

    def test_controlled_lcp_disabled_checkpoint_baseline_succeeds(self):
        args = Namespace(
            lcp_prefix_tokens=4,
            lcp_suffix_tokens=2,
            repeat_prefix_tokens=0,
            repeat_suffix_tokens=0,
            checkpoint_store_prefix_lengths=None,
            checkpoint_frontier_min_tokens=8192,
            no_prompt_checkpoint=True,
            target_tokens=None,
        )
        old_build_prompts = benchmark.build_controlled_lcp_prompts
        old_run_once = benchmark.run_once
        calls = []

        def fake_build_prompts(_tokenizer, prefix_tokens, suffix_tokens):
            requested = prefix_tokens + suffix_tokens
            return list(range(requested)), list(range(requested))

        def fake_run_once(_model, _tokenizer, _prompt, _args, case_name):
            calls.append(case_name)
            return {
                "case": case_name,
                "checkpoint_resolution": "disabled",
                "disk_cached_tokens": 0,
                "fresh_prompt_tokens": 6,
                "fresh_prefill_tokens": 5,
            }

        try:
            benchmark.build_controlled_lcp_prompts = fake_build_prompts
            benchmark.run_once = fake_run_once
            rows = benchmark.run_controlled_lcp(None, None, args)
        finally:
            benchmark.build_controlled_lcp_prompts = old_build_prompts
            benchmark.run_once = old_run_once

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["case"], "controlled-lcp-hit")
        self.assertEqual(rows[-1]["checkpoint_resolution"], "disabled")
        self.assertEqual(rows[-1]["stored_prefix_tokens"], 0)
        self.assertEqual(rows[-1]["expected_reused_prefix_tokens"], 0)
        self.assertEqual(rows[-1]["disk_cached_tokens"], 0)
        self.assertTrue(rows[-1]["checkpoint_expected_match"])
        self.assertEqual(rows[-1]["fresh_prompt_tokens"], 6)
        self.assertEqual(rows[-1]["fresh_prefill_tokens"], 5)

    def test_controlled_lcp_disabled_checkpoint_writes_json_output(self):
        old_argv = sys.argv
        old_load = benchmark.load
        old_build_prompts = benchmark.build_controlled_lcp_prompts
        old_run_once = benchmark.run_once

        def fake_load(*_args, **_kwargs):
            return object(), object()

        def fake_build_prompts(_tokenizer, prefix_tokens, suffix_tokens):
            requested = prefix_tokens + suffix_tokens
            return list(range(requested)), list(range(requested))

        def fake_run_once(_model, _tokenizer, _prompt, _args, case_name):
            return {
                "case": case_name,
                "checkpoint_resolution": "disabled",
                "disk_cached_tokens": 0,
                "fresh_prompt_tokens": 6,
                "fresh_prefill_tokens": 5,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, "controlled-lcp-disabled.json")
            sys.argv = [
                "glm52_prefill_benchmark.py",
                "--model",
                "dummy-model",
                "--mode",
                "controlled-lcp",
                "--lcp-prefix-tokens",
                "4",
                "--lcp-suffix-tokens",
                "2",
                "--no-prompt-checkpoint",
                "--json-output",
                json_output,
            ]
            benchmark.load = fake_load
            benchmark.build_controlled_lcp_prompts = fake_build_prompts
            benchmark.run_once = fake_run_once
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main()
            finally:
                sys.argv = old_argv
                benchmark.load = old_load
                benchmark.build_controlled_lcp_prompts = old_build_prompts
                benchmark.run_once = old_run_once

            with open(json_output, "r", encoding="utf-8") as f:
                payload = json.load(f)

        hit_row = payload["summary"][-1]
        self.assertEqual(hit_row["checkpoint_resolution"], "disabled")
        self.assertEqual(hit_row["expected_reused_prefix_tokens"], 0)
        self.assertEqual(hit_row["disk_cached_tokens"], 0)
        self.assertTrue(hit_row["checkpoint_expected_match"])


if __name__ == "__main__":
    unittest.main()
