import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

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
                    "prefill_max_qk_tokens=67108864 "
                    "glm_dsa_adaptive_prefill_step_size=8192 "
                    "glm_dsa_adaptive_prefill_after_tokens=4096 "
                    "glm_dsa_adaptive_prefill_min_remaining_tokens=2048 "
                    "resolution=prefix files_scanned=1 candidates_scanned=1 "
                    "matched_candidates=1 manifest_entries=1 "
                    "manifest_bootstrap=0 lookup_seconds=0.001234"
                ),
                (
                    "prompt checkpoint: prefill chunk start_tokens=6144 "
                    "chunk_tokens=2048 processed_tokens=8192 "
                    "total_prompt_tokens=8192 prefill_step_size=2048 "
                    "adaptive_prefill_step_size=8192 "
                    "effective_prefill_step_size=2048 "
                    "prefill_max_qk_tokens=67108864 chunk_seconds=1.000000"
                ),
            ]
        )

        self.assertEqual(summary["checkpoint_total_prompt_tokens"], 8192)
        self.assertEqual(summary["disk_cached_tokens"], 6144)
        self.assertEqual(summary["fresh_prompt_tokens"], 2048)
        self.assertEqual(summary["fresh_prefill_tokens"], 2047)
        self.assertEqual(summary["checkpoint_resolution"], "prefix")
        self.assertEqual(summary["checkpoint_lookup_seconds"], 0.001234)
        self.assertEqual(summary["checkpoint_prefill_max_qk_tokens"], 67108864)
        self.assertEqual(
            summary["checkpoint_glm_dsa_adaptive_prefill_step_size"], 8192
        )
        self.assertEqual(
            summary["checkpoint_glm_dsa_adaptive_prefill_after_tokens"], 4096
        )
        self.assertEqual(
            summary["checkpoint_glm_dsa_adaptive_prefill_min_remaining_tokens"],
            2048,
        )
        self.assertEqual(summary["checkpoint_prefill_chunks"], 1)
        self.assertEqual(summary["checkpoint_max_adaptive_prefill_step_size"], 8192)
        self.assertEqual(summary["checkpoint_max_effective_prefill_step_size"], 2048)

    def test_format_output_cell_serializes_compound_values(self):
        self.assertEqual(benchmark.format_output_cell({"decode": 156}), '{"decode":156}')
        self.assertEqual(benchmark.format_output_cell(None), "")

    def test_main_rejects_empty_model_argument(self):
        old_argv = sys.argv
        sys.argv = [
            "glm52_prefill_benchmark.py",
            "--model",
            "",
        ]
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as cm:
                    benchmark.main()
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--model is empty", stderr.getvalue())
        finally:
            sys.argv = old_argv

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

    def test_policy_sweep_default_candidates_include_ds4_boundary(self):
        args = Namespace(
            policy_candidates=None,
            policy_min_tokens=512,
            policy_boundary_trim_tokens=32,
            policy_boundary_align_tokens=2048,
        )

        candidates = benchmark.policy_sweep_candidates(
            args,
            prefix_tokens=3000,
            requested_total_tokens=4096,
        )

        self.assertEqual(
            candidates,
            [("disabled", 0), ("ds4-boundary", 2048), ("full-prefix", 3000)],
        )

    def test_prefill_sweep_default_candidates_compare_adaptive(self):
        args = Namespace(
            prefill_step_candidates=None,
            prefill_max_qk_token_candidates=None,
            prefill_max_qk_tokens=67_108_864,
            glm_dsa_adaptive_prefill_step_candidates=None,
            fast_prefill_min_context=None,
            fast_prefill_min_context_candidates=None,
        )

        candidates = benchmark.prefill_sweep_candidates(args)

        self.assertEqual(len(candidates), 6)
        self.assertEqual(candidates[0]["prefill_step_size"], 512)
        self.assertEqual(candidates[0]["prefill_max_qk_tokens"], 67_108_864)
        self.assertEqual(candidates[0]["glm_dsa_adaptive_prefill_step_size"], 0)
        self.assertIsNone(candidates[0]["fast_prefill_min_context"])
        self.assertEqual(candidates[1]["glm_dsa_adaptive_prefill_step_size"], 8192)

    def test_prefill_sweep_disables_checkpoints_by_default(self):
        args = Namespace(
            prompt_file=None,
            lengths="8",
            repeat_runs=1,
            prefill_step_candidates=[512, 1024],
            prefill_max_qk_token_candidates=[0],
            prefill_max_qk_tokens=67_108_864,
            glm_dsa_adaptive_prefill_step_candidates=[0, 8192],
            glm_dsa_adaptive_prefill_step_size=0,
            prefill_step_size=2048,
            no_prompt_checkpoint=False,
            checkpoint_save_exact="enabled",
            prefill_sweep_use_checkpoints=False,
            target_tokens=None,
            json_output=None,
            fast_prefill_min_context=None,
            fast_prefill_min_context_candidates=None,
        )
        old_build_prompt_text = benchmark.build_prompt_text
        old_run_once = benchmark.run_once
        calls = []

        def fake_build_prompt_text(_tokenizer, target_tokens, prefix_text=""):
            return f"prompt-{target_tokens}-{prefix_text}"

        def fake_run_once(_model, _tokenizer, _prompt, call_args, case_name):
            calls.append(
                {
                    "case": case_name,
                    "step": call_args.prefill_step_size,
                    "qk": call_args.prefill_max_qk_tokens,
                    "adaptive": call_args.glm_dsa_adaptive_prefill_step_size,
                    "min_context": call_args.fast_prefill_min_context,
                    "no_checkpoint": call_args.no_prompt_checkpoint,
                    "save_exact": call_args.checkpoint_save_exact,
                }
            )
            return {"case": case_name, "mode": "single"}

        try:
            benchmark.build_prompt_text = fake_build_prompt_text
            benchmark.run_once = fake_run_once
            rows = benchmark.run_prefill_sweep(None, None, args)
        finally:
            benchmark.build_prompt_text = old_build_prompt_text
            benchmark.run_once = old_run_once

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(call["no_checkpoint"] for call in calls))
        self.assertTrue(all(call["save_exact"] == "disabled" for call in calls))
        self.assertEqual(calls[0]["step"], 512)
        self.assertEqual(calls[1]["adaptive"], 8192)
        self.assertIsNone(calls[0]["min_context"])
        self.assertEqual(rows[0]["mode"], "prefill-sweep")
        self.assertEqual(rows[0]["prefill_sweep_candidate_index"], 0)

    def test_prefill_sweep_sweeps_fast_prefill_min_context(self):
        args = Namespace(
            prompt_file=None,
            lengths="8",
            repeat_runs=1,
            prefill_step_candidates=[2048],
            prefill_max_qk_token_candidates=[67_108_864],
            prefill_max_qk_tokens=67_108_864,
            glm_dsa_adaptive_prefill_step_candidates=[0],
            glm_dsa_adaptive_prefill_step_size=0,
            prefill_step_size=2048,
            no_prompt_checkpoint=False,
            checkpoint_save_exact="enabled",
            prefill_sweep_use_checkpoints=False,
            target_tokens=None,
            json_output=None,
            fast_prefill_min_context=None,
            fast_prefill_min_context_candidates=[98_304, 131_072],
        )
        env_key = benchmark.glm_moe_dsa.GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV
        saved_env = os.environ.get(env_key)
        old_build_prompt_text = benchmark.build_prompt_text
        old_run_once = benchmark.run_once
        calls = []

        def fake_build_prompt_text(_tokenizer, target_tokens, prefix_text=""):
            return f"prompt-{target_tokens}-{prefix_text}"

        def fake_run_once(_model, _tokenizer, _prompt, call_args, case_name):
            calls.append(
                {
                    "case": case_name,
                    "min_context": call_args.fast_prefill_min_context,
                    "env": os.environ.get(env_key),
                }
            )
            return {"case": case_name, "mode": "single"}

        try:
            os.environ[env_key] = "999"
            benchmark.build_prompt_text = fake_build_prompt_text
            benchmark.run_once = fake_run_once
            rows = benchmark.run_prefill_sweep(None, None, args)
            restored_env = os.environ.get(env_key)
        finally:
            benchmark.build_prompt_text = old_build_prompt_text
            benchmark.run_once = old_run_once
            if saved_env is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = saved_env

        self.assertEqual(len(rows), 2)
        self.assertEqual([call["min_context"] for call in calls], [98_304, 131_072])
        self.assertEqual([call["env"] for call in calls], ["98304", "131072"])
        self.assertEqual(restored_env, "999")
        self.assertEqual(rows[0]["prefill_sweep_fast_prefill_min_context"], 98_304)

    def test_prefill_sweep_writes_partial_json_output(self):
        old_build_prompt_text = benchmark.build_prompt_text
        old_run_once = benchmark.run_once

        def fake_build_prompt_text(_tokenizer, target_tokens, prefix_text=""):
            return f"prompt-{target_tokens}-{prefix_text}"

        def fake_run_once(_model, _tokenizer, _prompt, _args, case_name):
            return {
                "case": case_name,
                "mode": "single",
                "ttft_seconds": 1.0,
                "prompt_tps": 2.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = Path(tmpdir) / "prefill-sweep.json"
            args = Namespace(
                prompt_file=None,
                lengths="8",
                repeat_runs=1,
                prefill_step_candidates=[512],
                prefill_max_qk_token_candidates=[0],
                prefill_max_qk_tokens=0,
                glm_dsa_adaptive_prefill_step_candidates=[0],
                glm_dsa_adaptive_prefill_step_size=0,
                prefill_step_size=2048,
                no_prompt_checkpoint=False,
                checkpoint_save_exact="enabled",
                prefill_sweep_use_checkpoints=False,
                target_tokens=None,
                json_output=json_output,
                fast_prefill_min_context=None,
                fast_prefill_min_context_candidates=None,
            )
            benchmark.build_prompt_text = fake_build_prompt_text
            benchmark.run_once = fake_run_once
            try:
                rows = benchmark.run_prefill_sweep(None, None, args)
            finally:
                benchmark.build_prompt_text = old_build_prompt_text
                benchmark.run_once = old_run_once

            partial_output = benchmark.partial_json_output_path(json_output)
            with open(partial_output, "r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertEqual(len(rows), 1)
        self.assertTrue(payload["partial"])
        self.assertEqual(payload["completed_runs"], 1)
        self.assertEqual(payload["runs"][0]["mode"], "prefill-sweep")
        self.assertEqual(payload["summary"][0]["case"], rows[0]["case"])

    def test_run_once_can_stop_after_prefill_tokens(self):
        old_stream_generate = benchmark.stream_generate

        def fake_stream_generate(**kwargs):
            progress = kwargs["prompt_progress_callback"]
            progress(0, 10)
            progress(6, 10)
            yield None

        args = Namespace(
            target_tokens=10,
            max_tokens=1,
            prefill_step_size=4,
            prefill_max_qk_tokens=67_108_864,
            glm_dsa_adaptive_prefill_step_size=0,
            glm_dsa_adaptive_prefill_after_tokens=0,
            glm_dsa_adaptive_prefill_min_remaining_tokens=0,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=0,
            no_prompt_checkpoint=True,
            checkpoint_store_prefix_lengths=None,
            checkpoint_save_exact="disabled",
            checkpoint_frontier_min_tokens=8192,
            checkpoint_frontier_stride_tokens=16384,
            resolved_checkpoint_cache_dir=None,
            prefill_stop_after_tokens=6,
            prefill_profile=False,
            fast_prefill="enabled",
        )

        benchmark.stream_generate = fake_stream_generate
        try:
            row = benchmark.run_once(
                object(),
                object(),
                list(range(10)),
                args,
                "early-stop",
            )
        finally:
            benchmark.stream_generate = old_stream_generate

        self.assertTrue(row["prefill_stopped_early"])
        self.assertEqual(row["finish_reason"], "prefill-stop-after-tokens")
        self.assertEqual(row["partial_prefill_tokens"], 6)
        self.assertEqual(row["partial_prefill_total_tokens"], 10)
        self.assertEqual(row["prefill_stop_after_tokens"], 6)
        self.assertGreater(row["prompt_tps"], 0)

    def test_collect_profile_reports_native_sparse_prefill_status(self):
        old_profile = benchmark.glm_moe_dsa.get_glm_dsa_prefill_profile
        old_status = benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status
        old_indexer_status = (
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status
        )
        old_q8_status = benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status
        old_q4_qa_status = benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qa_status
        old_q4_status = benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qb_status

        def fake_profile():
            return {
                "stages": {},
                "fast_prefill_hits": 3,
                "fallback_reasons": {"below_sparse_min_context": 1},
                "native_sparse_prefill_hits": 2,
                "native_sparse_prefill_fallback_reasons": {"quantized_kv": 1},
                "native_indexer_hits": 5,
                "native_indexer_fallback_reasons": {
                    "below_native_indexer_min_context": 1
                },
                "native_q8_vup_hits": 4,
                "native_q8_vup_fallback_reasons": {"unsupported_heads:1": 1},
                "native_q4_qa_hits": 7,
                "native_q4_qa_fallback_reasons": {"missing_symbol": 1},
                "native_q4_qb_hits": 6,
                "native_q4_qb_fallback_reasons": {"disabled": 1},
            }

        def fake_status():
            return {
                "enabled": True,
                "available": True,
                "source": "test",
                "import_error": None,
                "min_context": 0,
            }

        def fake_indexer_status():
            return {
                "enabled": True,
                "available": True,
                "source": "indexer-test",
                "import_error": None,
                "scores_available": True,
                "topk_available": True,
                "min_context": 4096,
            }

        def fake_q8_status():
            return {
                "enabled": True,
                "available": True,
                "source": "q8-test",
                "import_error": None,
            }

        def fake_q4_status():
            return {
                "enabled": True,
                "available": True,
                "source": "q4-test",
                "import_error": None,
            }

        def fake_q4_qa_status():
            return {
                "enabled": True,
                "available": True,
                "source": "q4-qa-test",
                "import_error": None,
            }

        args = Namespace(
            prefill_profile=False,
            fast_prefill="enabled",
            native_sparse_prefill="enabled",
            native_indexer="enabled",
            native_q8_vup="enabled",
            native_q4_qa="enabled",
            native_q4_qb="enabled",
        )
        benchmark.glm_moe_dsa.get_glm_dsa_prefill_profile = fake_profile
        benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = fake_status
        benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
            fake_indexer_status
        )
        benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status = fake_q8_status
        benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qa_status = fake_q4_qa_status
        benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qb_status = fake_q4_status
        try:
            profile = benchmark.collect_glm_dsa_profile(args)
        finally:
            benchmark.glm_moe_dsa.get_glm_dsa_prefill_profile = old_profile
            benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = old_status
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
                old_indexer_status
            )
            benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status = old_q8_status
            benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qa_status = (
                old_q4_qa_status
            )
            benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qb_status = old_q4_status

        self.assertEqual(profile["glm_dsa_native_sparse_prefill"], "enabled")
        self.assertTrue(profile["glm_dsa_native_sparse_prefill_available"])
        self.assertEqual(profile["glm_dsa_native_sparse_prefill_source"], "test")
        self.assertEqual(profile["glm_dsa_native_sparse_prefill_hits"], 2)
        self.assertEqual(
            profile["glm_dsa_native_sparse_prefill_fallback_reasons"],
            {"quantized_kv": 1},
        )
        self.assertEqual(
            profile["glm_dsa_native_sparse_prefill_route_state"],
            "hit_with_fallbacks",
        )
        self.assertEqual(
            profile["glm_dsa_native_sparse_prefill_primary_fallback"],
            "quantized_kv",
        )
        self.assertIsNone(profile["glm_dsa_native_sparse_prefill_config_blocker"])
        self.assertEqual(profile["glm_dsa_native_indexer"], "enabled")
        self.assertTrue(profile["glm_dsa_native_indexer_available"])
        self.assertEqual(profile["glm_dsa_native_indexer_source"], "indexer-test")
        self.assertEqual(profile["glm_dsa_native_indexer_hits"], 5)
        self.assertEqual(
            profile["glm_dsa_native_indexer_fallback_reasons"],
            {"below_native_indexer_min_context": 1},
        )
        self.assertEqual(profile["glm_dsa_native_q8_vup"], "enabled")
        self.assertTrue(profile["glm_dsa_native_q8_vup_available"])
        self.assertEqual(profile["glm_dsa_native_q8_vup_source"], "q8-test")
        self.assertEqual(profile["glm_dsa_native_q8_vup_hits"], 4)
        self.assertEqual(
            profile["glm_dsa_native_q8_vup_fallback_reasons"],
            {"unsupported_heads:1": 1},
        )
        self.assertEqual(profile["glm_dsa_native_q4_qa"], "enabled")
        self.assertTrue(profile["glm_dsa_native_q4_qa_available"])
        self.assertEqual(profile["glm_dsa_native_q4_qa_source"], "q4-qa-test")
        self.assertEqual(profile["glm_dsa_native_q4_qa_hits"], 7)
        self.assertEqual(
            profile["glm_dsa_native_q4_qa_fallback_reasons"],
            {"missing_symbol": 1},
        )
        self.assertEqual(profile["glm_dsa_native_q4_qb"], "enabled")
        self.assertTrue(profile["glm_dsa_native_q4_qb_available"])
        self.assertEqual(profile["glm_dsa_native_q4_qb_source"], "q4-test")
        self.assertEqual(profile["glm_dsa_native_q4_qb_hits"], 6)
        self.assertEqual(
            profile["glm_dsa_native_q4_qb_fallback_reasons"],
            {"disabled": 1},
        )

    def test_collect_profile_reports_quantized_native_route_blocker(self):
        old_profile = benchmark.glm_moe_dsa.get_glm_dsa_prefill_profile
        old_status = benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status
        old_indexer_status = (
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status
        )
        old_q8_status = benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status
        old_q4_qa_status = benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qa_status
        old_q4_status = benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qb_status
        env_key = benchmark.glm_moe_dsa.GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV
        quantized_env_key = (
            benchmark.glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV
        )
        old_env = os.environ.get(env_key)
        old_quantized_env = os.environ.get(quantized_env_key)

        def fake_profile():
            return {
                "stages": {},
                "fast_prefill_hits": 0,
                "fallback_reasons": {},
                "native_sparse_prefill_hits": 0,
                "native_sparse_prefill_fallback_reasons": {},
                "native_indexer_hits": 0,
                "native_indexer_fallback_reasons": {},
                "native_q8_vup_hits": 0,
                "native_q8_vup_fallback_reasons": {},
                "native_q4_qa_hits": 0,
                "native_q4_qa_fallback_reasons": {},
                "native_q4_qb_hits": 0,
                "native_q4_qb_fallback_reasons": {},
            }

        def fake_status():
            return {
                "enabled": True,
                "available": True,
                "source": "mlx_lm.custom_kernels.glm_moe_dsa",
                "import_error": None,
                "min_context": 11264,
            }

        def fake_indexer_status():
            return {
                "enabled": True,
                "available": True,
                "source": "indexer-test",
                "import_error": None,
                "scores_available": True,
                "topk_available": True,
                "min_context": 4096,
            }

        def fake_q8_status():
            return {
                "enabled": True,
                "available": True,
                "source": "q8-test",
                "import_error": None,
            }

        def fake_q4_status():
            return {
                "enabled": False,
                "available": True,
                "source": "q4-test",
                "import_error": None,
            }

        def fake_q4_qa_status():
            return {
                "enabled": False,
                "available": True,
                "source": "q4-qa-test",
                "import_error": None,
            }

        args = Namespace(
            prefill_profile=False,
            fast_prefill="enabled",
            native_sparse_prefill="enabled",
            native_indexer="enabled",
            native_q8_vup="enabled",
            native_q4_qa="default",
            native_q4_qb="default",
            mode="single",
            batch_size=1,
            kv_bits=8,
            quantized_kv_start=4096,
        )
        benchmark.glm_moe_dsa.get_glm_dsa_prefill_profile = fake_profile
        benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = fake_status
        benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
            fake_indexer_status
        )
        benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status = fake_q8_status
        benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qa_status = fake_q4_qa_status
        benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qb_status = fake_q4_status
        os.environ.pop(env_key, None)
        os.environ.pop(quantized_env_key, None)
        try:
            profile = benchmark.collect_glm_dsa_profile(args)
        finally:
            benchmark.glm_moe_dsa.get_glm_dsa_prefill_profile = old_profile
            benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = old_status
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
                old_indexer_status
            )
            benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status = old_q8_status
            benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qa_status = (
                old_q4_qa_status
            )
            benchmark.glm_moe_dsa.get_glm_dsa_native_q4_qb_status = old_q4_status
            if old_env is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old_env
            if old_quantized_env is None:
                os.environ.pop(quantized_env_key, None)
            else:
                os.environ[quantized_env_key] = old_quantized_env

        self.assertEqual(
            profile["glm_dsa_native_sparse_prefill_route_state"],
            "not_attempted",
        )
        self.assertEqual(
            profile["glm_dsa_native_sparse_prefill_config_blocker"],
            "quantized_kv_at_native_threshold",
        )
        self.assertEqual(
            profile["glm_dsa_native_sparse_prefill_attempt_min_context"],
            11264,
        )

    def test_native_sparse_config_blocker_allows_quantized_kv_opt_in(self):
        env_key = (
            benchmark.glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV
        )
        old_env = os.environ.get(env_key)
        args = Namespace(
            fast_prefill="enabled",
            native_sparse_prefill="enabled",
            mode="single",
            batch_size=1,
            kv_bits=8,
            quantized_kv_start=4096,
        )
        native_status = {
            "enabled": True,
            "available": True,
            "min_context": 11264,
        }

        os.environ[env_key] = "1"
        try:
            blocker = benchmark.native_sparse_prefill_config_blocker(
                args,
                native_status,
            )
        finally:
            if old_env is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old_env

        self.assertIsNone(blocker)

    def test_prefill_config_summary_reports_kv_quantization_settings(self):
        args = Namespace(
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=4096,
            prefill_step_size=1024,
            prefill_max_qk_tokens=67_108_864,
            glm_dsa_adaptive_prefill_step_size=0,
            glm_dsa_adaptive_prefill_after_tokens=0,
            glm_dsa_adaptive_prefill_min_remaining_tokens=0,
        )

        summary = benchmark.prefill_config_summary(args)

        self.assertEqual(summary["kv_bits"], 8)
        self.assertEqual(summary["kv_group_size"], 64)
        self.assertEqual(summary["quantized_kv_start"], 4096)

    def test_native_kernel_smoke_reports_unavailable_status(self):
        old_status = benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status
        old_indexer_status = (
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status
        )

        def fake_status():
            return {
                "enabled": True,
                "available": False,
                "source": None,
                "import_error": "ImportError('missing')",
                "min_context": 11264,
            }

        def fake_indexer_status():
            return {
                "enabled": True,
                "available": False,
                "source": None,
                "import_error": "ImportError('missing indexer')",
                "scores_available": False,
                "topk_available": False,
                "min_context": 4096,
            }

        args = Namespace(
            native_sparse_prefill="default",
            native_smoke_q_len=2,
            native_smoke_k_len=32,
            native_smoke_seed=7,
            native_smoke_max_diff=0.02,
        )
        benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = fake_status
        benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
            fake_indexer_status
        )
        try:
            row = benchmark.run_native_kernel_smoke(args)
        finally:
            benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = old_status
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
                old_indexer_status
            )

        self.assertFalse(row["native_smoke_available"])
        self.assertFalse(row["native_smoke_passed"])
        self.assertEqual(row["native_smoke_import_error"], "ImportError('missing')")
        self.assertEqual(
            row["native_smoke_error"],
            "native sparse MLA kernel unavailable",
        )
        self.assertEqual(row["glm_dsa_native_sparse_prefill_min_context"], 11264)
        self.assertFalse(row["native_indexer_smoke_available"])
        self.assertEqual(
            row["native_indexer_smoke_error"],
            "native DSA indexer kernels unavailable",
        )

    def test_native_q8_vup_smoke_benchmark_reports_timings(self):
        old_sparse_status = (
            benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status
        )
        old_indexer_status = (
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status
        )
        old_q8_status = benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status
        old_q8_kernel = benchmark._native_q8_vup_smoke_kernel

        def fake_sparse_status():
            return {
                "enabled": True,
                "available": False,
                "source": None,
                "import_error": "missing",
                "min_context": 11264,
            }

        def fake_indexer_status():
            return {
                "enabled": True,
                "available": False,
                "source": None,
                "import_error": "missing-indexer",
                "scores_available": False,
                "topk_available": False,
                "min_context": 4096,
            }

        def fake_q8_status():
            return {
                "enabled": True,
                "available": True,
                "source": "q8-test",
                "import_error": None,
            }

        def fake_q8_kernel(_source):
            def kernel(x, q_weight, scales, biases):
                return benchmark._native_q8_vup_reference(
                    x,
                    q_weight,
                    scales,
                    biases,
                )

            return kernel

        args = Namespace(
            native_sparse_prefill="default",
            native_smoke_q_len=2,
            native_smoke_k_len=32,
            native_smoke_seed=7,
            native_smoke_max_diff=0.02,
            native_smoke_benchmark_runs=1,
            native_smoke_benchmark_warmup_runs=0,
            native_q8_vup_benchmark_q_len=2,
        )
        benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = (
            fake_sparse_status
        )
        benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
            fake_indexer_status
        )
        benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status = fake_q8_status
        benchmark._native_q8_vup_smoke_kernel = fake_q8_kernel
        try:
            row = benchmark.run_native_kernel_smoke(args)
        finally:
            benchmark.glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status = (
                old_sparse_status
            )
            benchmark.glm_moe_dsa.get_glm_dsa_native_indexer_status = (
                old_indexer_status
            )
            benchmark.glm_moe_dsa.get_glm_dsa_native_q8_vup_status = old_q8_status
            benchmark._native_q8_vup_smoke_kernel = old_q8_kernel

        self.assertTrue(row["native_q8_vup_smoke_passed"])
        self.assertEqual(row["native_q8_vup_benchmark_runs"], 1)
        self.assertEqual(row["native_q8_vup_benchmark_q_len"], 2)
        self.assertIsNone(row["native_q8_vup_benchmark_error"])
        self.assertGreater(row["native_q8_vup_native_seconds_mean"], 0)
        self.assertGreater(row["native_q8_vup_reference_seconds_mean"], 0)
        self.assertIsNotNone(row["native_q8_vup_speedup_mean"])

    def test_main_native_smoke_does_not_load_model(self):
        old_argv = sys.argv
        old_load = benchmark.load
        old_run_native_kernel_smoke = benchmark.run_native_kernel_smoke
        calls = []

        def fake_load(*_args, **_kwargs):
            raise AssertionError("native-smoke should not load a model")

        def fake_run_native_kernel_smoke(args):
            calls.append(args.model)
            return {
                "case": "native-smoke",
                "mode": "native-smoke",
                "native_smoke_passed": True,
            }

        sys.argv = ["glm52_prefill_benchmark.py", "--mode", "native-smoke"]
        benchmark.load = fake_load
        benchmark.run_native_kernel_smoke = fake_run_native_kernel_smoke
        try:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                benchmark.main()
        finally:
            sys.argv = old_argv
            benchmark.load = old_load
            benchmark.run_native_kernel_smoke = old_run_native_kernel_smoke

        self.assertEqual(calls, [""])
        self.assertIn("native_smoke_passed", stdout.getvalue())
        self.assertIn("True", stdout.getvalue())

    def test_policy_sweep_runs_isolated_candidates(self):
        args = Namespace(
            lcp_prefix_tokens=4,
            lcp_suffix_tokens=2,
            repeat_prefix_tokens=0,
            repeat_suffix_tokens=0,
            policy_candidates=[("disabled", 0), ("p4", 4)],
            policy_min_tokens=1,
            policy_boundary_trim_tokens=0,
            policy_boundary_align_tokens=1,
            resolved_checkpoint_cache_dir=None,
            target_tokens=None,
            checkpoint_store_prefix_lengths=None,
            checkpoint_frontier_min_tokens=8192,
            no_prompt_checkpoint=False,
        )
        old_build_prompts = benchmark.build_controlled_lcp_prompts
        old_run_once = benchmark.run_once
        calls = []

        def fake_build_prompts(_tokenizer, prefix_tokens, suffix_tokens):
            requested = prefix_tokens + suffix_tokens
            return list(range(requested)), list(range(requested))

        def fake_run_once(_model, _tokenizer, _prompt, call_args, case_name):
            store_lengths = call_args.checkpoint_store_prefix_lengths or []
            disk_cached_tokens = 0
            if case_name.endswith("-hit") and not call_args.no_prompt_checkpoint:
                disk_cached_tokens = store_lengths[0]
            calls.append(
                {
                    "case": case_name,
                    "cache_dir": os.environ.get(
                        benchmark.prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV
                    ),
                    "no_prompt_checkpoint": call_args.no_prompt_checkpoint,
                    "store_lengths": store_lengths,
                }
            )
            return {
                "case": case_name,
                "mode": "single",
                "disk_cached_tokens": disk_cached_tokens,
                "fresh_prefill_tokens": 5 - disk_cached_tokens,
                "checkpoint_resolution": (
                    "disabled" if call_args.no_prompt_checkpoint else "prefix"
                ),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            args.resolved_checkpoint_cache_dir = tmpdir
            benchmark.build_controlled_lcp_prompts = fake_build_prompts
            benchmark.run_once = fake_run_once
            try:
                rows = benchmark.run_policy_sweep(None, None, args)
            finally:
                benchmark.build_controlled_lcp_prompts = old_build_prompts
                benchmark.run_once = old_run_once

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["policy_name"], "disabled")
        self.assertEqual(rows[1]["mode"], "policy-sweep")
        self.assertEqual(rows[1]["expected_reused_prefix_tokens"], 0)
        self.assertTrue(rows[1]["checkpoint_expected_match"])
        self.assertEqual(rows[3]["policy_name"], "p4")
        self.assertEqual(rows[3]["expected_reused_prefix_tokens"], 4)
        self.assertTrue(rows[3]["checkpoint_expected_match"])
        self.assertEqual(calls[0]["cache_dir"], calls[1]["cache_dir"])
        self.assertEqual(calls[2]["cache_dir"], calls[3]["cache_dir"])
        self.assertNotEqual(calls[1]["cache_dir"], calls[2]["cache_dir"])
        self.assertTrue(calls[0]["no_prompt_checkpoint"])
        self.assertEqual(calls[2]["store_lengths"], [4])


if __name__ == "__main__":
    unittest.main()
