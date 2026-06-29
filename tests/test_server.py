# Copyright © 2024 Apple Inc.

import http
import io
import json
import os
import tempfile
import threading
import types
import unittest
import unittest.mock as mock
from pathlib import Path
from queue import Queue

import mlx.core as mx
import requests

from mlx_lm.models.cache import KVCache, PROMPT_CHECKPOINT_CACHE_DIR_ENV
from mlx_lm.server import (
    APIHandler,
    DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS,
    DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS,
    DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS,
    DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS,
    DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS,
    DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS,
    DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT,
    LRUPromptCache,
    Response,
    ResponseGenerator,
    TokenLoopGuard,
    _prompt_checkpoint_boundary_store_length,
    _prompt_checkpoint_continued_frontier_args,
    _prompt_checkpoint_continued_store_length,
    _prompt_checkpoint_store_prefix_lengths,
    _process_control_tokens,
    configure_checkpoint_cache_dir,
    setup_arg_parser,
)
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prefill_step_size": 2048,
                "prefill_max_qk_tokens": 67_108_864,
                "checkpoint_min_tokens": DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS,
                "checkpoint_cold_max_tokens": (
                    DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS
                ),
                "checkpoint_boundary_trim_tokens": (
                    DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS
                ),
                "checkpoint_boundary_align_tokens": (
                    DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS
                ),
                "checkpoint_continued_interval_tokens": (
                    DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS
                ),
                "checkpoint_max_age_seconds": (
                    DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS
                ),
                "checkpoint_shutdown_save_limit": (
                    DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT
                ),
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
                "allowed_origins": ["*"],
                "disable_batching": False,
                "kv_bits": None,
                "kv_group_size": 64,
                "quantized_kv_start": 0,
                "loop_guard_ngram_size": 64,
                "loop_guard_repeats": 3,
                "loop_guard_min_tokens": 256,
                "decode_progress_interval_tokens": 0,
                "prefill_progress_interval_tokens": 0,
                "checkpoint_save_exact": "enabled",
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer

    def load_default(self):
        return self.load("default_model", None, "default_model")


class MockCache:
    def __init__(self, value, is_trimmable: bool = True):
        self.value = value
        self._is_trimmable = is_trimmable

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value

    def is_trimmable(self):
        return self._is_trimmable

    def trim(self, n):
        assert self._is_trimmable
        return n


class TestProcessControlTokens(unittest.TestCase):
    @staticmethod
    def _r(text, state, match=None):
        return Response(text, 0, state, match, 0.0, None, ())

    def test_single_tool_call_passes_body_with_open_and_close_crossings(self):
        r = self._r
        stream = [
            r("hi ", "normal"),
            r("<tool_call>", "tool", match=(0,)),
            r("body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r(" bye", "normal"),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "hi body bye")
        states = [t.state for t in out]
        self.assertEqual(sum(1 for a, b in zip(states, states[1:]) if a != b), 2)

    def test_back_to_back_tool_calls_emit_state_crossings(self):
        r = self._r
        stream = [
            r("<tool_call>", "tool", match=(0,)),
            r("call1_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r("<tool_call>", "tool", match=(0,)),
            r("call2_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "call1_bodycall2_body")
        states = [t.state for t in out]
        crossings = sum(
            1 for a, b in zip(states, states[1:]) if a == "tool" and b == "normal"
        )
        self.assertEqual(crossings, 2)

    def test_multi_token_match_preserves_order(self):
        r = self._r
        match = (10, 11, 12)
        stream = [
            r("body", "tool"),
            r("</", "tool"),
            r("tool", "tool"),
            r("_call>", "normal", match=match),
            r(" ok", "normal"),
        ]
        ctx = types.SimpleNamespace(sequences={match: "</tool_call>"})
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual([t.text for t in out], ["body", "", "", "", " ok"])
        self.assertEqual(
            [t.state for t in out],
            ["tool", "tool", "tool", "normal", "normal"],
        )


class TestTokenLoopGuard(unittest.TestCase):
    def test_detects_exact_repeated_ngram(self):
        guard = TokenLoopGuard(ngram_size=4, repeats=3, min_tokens=0)
        results = [guard.append(t) for t in [1, 2, 3, 4] * 3]

        self.assertFalse(any(results[:-1]))
        self.assertTrue(results[-1])

    def test_min_tokens_delays_detection(self):
        guard = TokenLoopGuard(ngram_size=4, repeats=3, min_tokens=16)
        tokens = [9, 8, 7, 6] + [1, 2, 3, 4] * 3
        results = [guard.append(t) for t in tokens]

        self.assertFalse(any(results[:-1]))
        self.assertTrue(results[-1])

    def test_disabled_when_ngram_size_is_zero(self):
        guard = TokenLoopGuard(ngram_size=0, repeats=3, min_tokens=0)

        self.assertFalse(any(guard.append(t) for t in [1, 2, 3, 4] * 5))


class TestPromptCheckpointPolicy(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "checkpoint_min_tokens": DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS,
            "checkpoint_cold_max_tokens": DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS,
            "checkpoint_boundary_trim_tokens": (
                DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS
            ),
            "checkpoint_boundary_align_tokens": (
                DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS
            ),
            "checkpoint_continued_interval_tokens": (
                DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS
            ),
            "checkpoint_max_age_seconds": DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS,
            "checkpoint_shutdown_save_limit": (
                DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT
            ),
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_boundary_store_length_matches_ds4_defaults(self):
        self.assertEqual(_prompt_checkpoint_boundary_store_length(11_011), 10_240)
        self.assertEqual(_prompt_checkpoint_boundary_store_length(1_695), 1_695)
        self.assertEqual(
            _prompt_checkpoint_boundary_store_length(
                3_500,
                min_tokens=512,
                trim_tokens=0,
                align_tokens=1_000,
            ),
            3_000,
        )
        self.assertEqual(
            _prompt_checkpoint_boundary_store_length(
                3_500,
                min_tokens=512,
                trim_tokens=0,
                align_tokens=0,
            ),
            3_500,
        )

    def test_store_prefix_lengths_add_system_and_cold_boundary(self):
        args = self._args()
        prompt = [0] * 11_011
        segments = [[0] * 600, [0] * (len(prompt) - 600)]

        lengths = _prompt_checkpoint_store_prefix_lengths(
            args,
            prompt,
            segments,
            ["system", "user"],
        )

        self.assertEqual(lengths, [600, 10_240])

    def test_store_prefix_lengths_skip_cached_and_too_long_cold(self):
        args = self._args()
        prompt = [0] * 40_000
        segments = [[0] * 600, [0] * (len(prompt) - 600)]

        lengths = _prompt_checkpoint_store_prefix_lengths(
            args,
            prompt,
            segments,
            ["system", "user"],
            initial_cached_tokens=700,
        )

        self.assertEqual(lengths, [])

    def test_continued_frontier_args_align_interval_up(self):
        self.assertEqual(
            _prompt_checkpoint_continued_frontier_args(self._args()),
            (10_240, 10_240),
        )
        self.assertEqual(
            _prompt_checkpoint_continued_frontier_args(
                self._args(checkpoint_boundary_align_tokens=0)
            ),
            (10_000, 10_000),
        )
        self.assertEqual(
            _prompt_checkpoint_continued_frontier_args(
                self._args(checkpoint_continued_interval_tokens=0)
            ),
            (0, 0),
        )

    def test_continued_store_length_uses_interval_boundary(self):
        args = self._args(
            checkpoint_min_tokens=1,
            checkpoint_boundary_trim_tokens=0,
            checkpoint_boundary_align_tokens=4,
            checkpoint_continued_interval_tokens=5,
        )

        self.assertEqual(_prompt_checkpoint_continued_store_length(args, 7), 0)
        self.assertEqual(_prompt_checkpoint_continued_store_length(args, 8), 8)
        self.assertEqual(_prompt_checkpoint_continued_store_length(args, 15), 8)
        self.assertEqual(_prompt_checkpoint_continued_store_length(args, 16), 16)

    def test_save_continued_prompt_checkpoint_trims_and_records_metadata(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        cli_args = self._args(
            checkpoint_min_tokens=1,
            checkpoint_boundary_trim_tokens=0,
            checkpoint_boundary_align_tokens=4,
            checkpoint_continued_interval_tokens=5,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        generator.model_provider = types.SimpleNamespace(
            model=object(),
            draft_model=None,
            cli_args=cli_args,
        )
        tokenizer = types.SimpleNamespace(
            decode=lambda tokens, **kwargs: "".join(f"<{token}>" for token in tokens)
        )
        cache_key = list(range(1, 12))
        prompt_cache = [MockCache("cache")]
        rendered_continuation = "".join(f"<{token}>" for token in cache_key)
        captured = {}

        def fake_save_prompt_checkpoint(file_name, cache, **kwargs):
            captured["file_name"] = file_name
            captured["cache"] = cache
            captured["kwargs"] = kwargs
            return dict(kwargs["metadata"])

        with mock.patch(
            "mlx_lm.server.save_prompt_checkpoint",
            fake_save_prompt_checkpoint,
        ), mock.patch(
            "mlx_lm.server.update_prompt_checkpoint_manifest"
        ) as update_manifest, mock.patch(
            "mlx_lm.server.prune_prompt_checkpoints"
        ) as prune:
            saved = generator._save_continued_prompt_checkpoint(
                tokenizer,
                prompt_cache,
                cache_key,
                prompt_token_count=5,
                rendered_continuation=rendered_continuation,
            )

        self.assertTrue(saved)
        self.assertEqual(captured["kwargs"]["prefix_tokens"], cache_key[:8])
        self.assertEqual(
            captured["kwargs"]["metadata"]["checkpoint_label"],
            "continued",
        )
        self.assertIn("checkpoint_prefix_tokens", captured["kwargs"]["metadata"])
        self.assertIn(
            "checkpoint_rendered_prefix_hash",
            captured["kwargs"]["metadata"],
        )
        self.assertIsNot(captured["cache"], prompt_cache)
        update_manifest.assert_called_once()
        self.assertEqual(update_manifest.call_args.kwargs["kind"], "continued")
        prune.assert_called_once()

    def test_shutdown_flush_saves_current_model_continued_checkpoint(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        cli_args = self._args(
            checkpoint_min_tokens=1,
            checkpoint_boundary_trim_tokens=0,
            checkpoint_boundary_align_tokens=4,
            checkpoint_continued_interval_tokens=5,
            checkpoint_shutdown_save_limit=1,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        tokenizer = types.SimpleNamespace(decode=lambda tokens, **kwargs: "")
        generator.model_provider = types.SimpleNamespace(
            model=object(),
            tokenizer=tokenizer,
            draft_model=None,
            model_key=("model", None, None),
            cli_args=cli_args,
        )
        generator.prompt_cache = LRUPromptCache(max_size=10)
        current_cache = [MockCache("current-cache")]
        other_cache = [MockCache("other-cache")]
        generator.prompt_cache.insert_cache(
            ("other", None, None),
            list(range(20)),
            other_cache,
        )
        generator.prompt_cache.insert_cache(
            generator.model_provider.model_key,
            list(range(12)),
            current_cache,
        )
        captured = {}

        def fake_save_prompt_checkpoint(file_name, cache, **kwargs):
            captured["file_name"] = file_name
            captured["cache"] = cache
            captured["kwargs"] = kwargs
            return dict(kwargs["metadata"])

        with mock.patch(
            "mlx_lm.server.save_prompt_checkpoint",
            fake_save_prompt_checkpoint,
        ), mock.patch(
            "mlx_lm.server.update_prompt_checkpoint_manifest"
        ), mock.patch(
            "mlx_lm.server.prune_prompt_checkpoints"
        ):
            stats = generator.flush_shutdown_prompt_checkpoints()

        self.assertEqual(stats["candidates"], 1)
        self.assertEqual(stats["attempted"], 1)
        self.assertEqual(stats["saved"], 1)
        self.assertEqual(captured["kwargs"]["prefix_tokens"], list(range(8)))
        self.assertIsNot(captured["cache"], current_cache)

    def test_render_and_tokenize_are_idempotent_for_tool_arguments(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            cli_args=types.SimpleNamespace(chat_template_args={})
        )

        class FakeChatTokenizer:
            has_chat_template = True
            has_tool_calling = True
            has_thinking = False
            tool_parser = None

            def apply_chat_template(
                self,
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **kwargs,
            ):
                text = json.dumps(messages, sort_keys=True)
                if add_generation_prompt:
                    text += "<gen>"
                return [ord(ch) for ch in text] if tokenize else text

        request = types.SimpleNamespace(
            request_type="chat",
            messages=[
                {"role": "user", "content": "call a tool"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call-1",
                            "function": {
                                "name": "search",
                                "arguments": '{"query":"prefill"}',
                            },
                        }
                    ],
                },
            ],
            tools=None,
            role_mapping=None,
        )
        args = types.SimpleNamespace(chat_template_kwargs=None)
        tokenizer = FakeChatTokenizer()

        rendered = generator._render_prompt_text(tokenizer, request, args)
        prompt, segments, segment_types, initial_state, tokenized_rendered = (
            generator._tokenize(tokenizer, request, args, return_rendered=True)
        )

        self.assertEqual(rendered, tokenized_rendered)
        self.assertEqual(prompt, [ord(ch) for ch in rendered])
        self.assertEqual(segments, [prompt])
        self.assertEqual(segment_types, ["assistant"])
        self.assertEqual(initial_state, "normal")
        arguments = request.messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, {"query": "prefill"})

    def test_rendered_checkpoint_rejects_prefix_decode_mismatch(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            model=object(),
            draft_model=None,
            cli_args=self._args(kv_bits=None, kv_group_size=64, quantized_kv_start=0),
        )
        tokenizer = types.SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [ord(ch) for ch in text],
            decode=lambda tokens, **kwargs: "ab" if tokens == [1, 2] else "",
        )

        with mock.patch(
            "mlx_lm.server.find_prompt_checkpoint_rendered_prefix",
            return_value=([(3, 2, "/tmp/frontier.safetensors", "frontier")], {}),
        ), mock.patch(
            "mlx_lm.server.load_prompt_checkpoint_with_metadata_prefix",
            return_value=(["cache"], [1, 2], {"checkpoint_label": "frontier"}),
        ), mock.patch(
            "mlx_lm.server.update_prompt_checkpoint_manifest"
        ) as update_manifest:
            result = generator._load_rendered_prompt_checkpoint(tokenizer, "abcXYZ")

        self.assertIsNone(result)
        update_manifest.assert_not_called()

    def test_rendered_checkpoint_rejects_suffix_decode_mismatch(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            model=object(),
            draft_model=None,
            cli_args=self._args(kv_bits=None, kv_group_size=64, quantized_kv_start=0),
        )

        def decode(tokens, **kwargs):
            if tokens == [1, 2]:
                return "abc"
            if tokens == [9]:
                return "not-XYZ"
            return ""

        tokenizer = types.SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [9],
            decode=decode,
        )

        with mock.patch(
            "mlx_lm.server.find_prompt_checkpoint_rendered_prefix",
            return_value=([(3, 2, "/tmp/frontier.safetensors", "frontier")], {}),
        ), mock.patch(
            "mlx_lm.server.load_prompt_checkpoint_with_metadata_prefix",
            return_value=(["cache"], [1, 2], {"checkpoint_label": "frontier"}),
        ), mock.patch(
            "mlx_lm.server.update_prompt_checkpoint_manifest"
        ) as update_manifest:
            result = generator._load_rendered_prompt_checkpoint(tokenizer, "abcXYZ")

        self.assertIsNone(result)
        update_manifest.assert_not_called()


class TestServerCLI(unittest.TestCase):
    def test_setup_arg_parser_accepts_kv_options(self):
        args = setup_arg_parser().parse_args(
            [
                "--kv-bits",
                "8",
                "--kv-group-size",
                "64",
                "--quantized-kv-start",
                "4096",
            ]
        )

        self.assertEqual(args.kv_bits, 8)
        self.assertEqual(args.kv_group_size, 64)
        self.assertEqual(args.quantized_kv_start, 4096)

    def test_setup_arg_parser_kv_defaults(self):
        args = setup_arg_parser().parse_args([])

        self.assertIsNone(args.kv_bits)
        self.assertEqual(args.kv_group_size, 64)
        self.assertEqual(args.quantized_kv_start, 0)
        self.assertFalse(args.disable_batching)
        self.assertEqual(args.prefill_max_qk_tokens, 67_108_864)
        self.assertIsNone(args.checkpoint_cache_dir)
        self.assertEqual(
            args.checkpoint_min_tokens, DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS
        )
        self.assertEqual(
            args.checkpoint_cold_max_tokens,
            DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS,
        )
        self.assertEqual(
            args.checkpoint_boundary_trim_tokens,
            DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS,
        )
        self.assertEqual(
            args.checkpoint_boundary_align_tokens,
            DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS,
        )
        self.assertEqual(
            args.checkpoint_continued_interval_tokens,
            DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS,
        )
        self.assertEqual(
            args.checkpoint_max_age_seconds,
            DEFAULT_PROMPT_CHECKPOINT_MAX_AGE_SECONDS,
        )
        self.assertEqual(
            args.checkpoint_shutdown_save_limit,
            DEFAULT_PROMPT_CHECKPOINT_SHUTDOWN_SAVE_LIMIT,
        )
        self.assertEqual(args.loop_guard_ngram_size, 64)
        self.assertEqual(args.loop_guard_repeats, 3)
        self.assertEqual(args.loop_guard_min_tokens, 256)
        self.assertEqual(args.decode_progress_interval_tokens, 0)
        self.assertEqual(args.prefill_progress_interval_tokens, 0)
        self.assertEqual(args.checkpoint_save_exact, "enabled")

    def test_setup_arg_parser_disable_batching(self):
        args = setup_arg_parser().parse_args(["--disable-batching"])

        self.assertTrue(args.disable_batching)

    def test_setup_arg_parser_prefill_max_qk_tokens(self):
        args = setup_arg_parser().parse_args(["--prefill-max-qk-tokens", "0"])

        self.assertEqual(args.prefill_max_qk_tokens, 0)

    def test_setup_arg_parser_checkpoint_cache_dir(self):
        args = setup_arg_parser().parse_args(
            ["--checkpoint-cache-dir", "/tmp/glm52-checkpoints"]
        )

        self.assertEqual(args.checkpoint_cache_dir, Path("/tmp/glm52-checkpoints"))

    def test_setup_arg_parser_checkpoint_policy_options(self):
        args = setup_arg_parser().parse_args(
            [
                "--checkpoint-min-tokens",
                "128",
                "--checkpoint-cold-max-tokens",
                "4096",
                "--checkpoint-boundary-trim-tokens",
                "16",
                "--checkpoint-boundary-align-tokens",
                "1024",
                "--checkpoint-continued-interval-tokens",
                "8192",
                "--checkpoint-max-age-seconds",
                "3600",
                "--checkpoint-save-exact",
                "disabled",
                "--checkpoint-shutdown-save-limit",
                "2",
            ]
        )

        self.assertEqual(args.checkpoint_min_tokens, 128)
        self.assertEqual(args.checkpoint_cold_max_tokens, 4096)
        self.assertEqual(args.checkpoint_boundary_trim_tokens, 16)
        self.assertEqual(args.checkpoint_boundary_align_tokens, 1024)
        self.assertEqual(args.checkpoint_continued_interval_tokens, 8192)
        self.assertEqual(args.checkpoint_max_age_seconds, 3600)
        self.assertEqual(args.checkpoint_save_exact, "disabled")
        self.assertEqual(args.checkpoint_shutdown_save_limit, 2)

    def test_setup_arg_parser_no_save_exact_checkpoint_alias(self):
        args = setup_arg_parser().parse_args(["--no-save-exact-checkpoint"])

        self.assertEqual(args.checkpoint_save_exact, "disabled")

    def test_configure_checkpoint_cache_dir_sets_env(self):
        old_value = os.environ.get(PROMPT_CHECKPOINT_CACHE_DIR_ENV)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "prompt-checkpoints"
                resolved = configure_checkpoint_cache_dir(
                    types.SimpleNamespace(checkpoint_cache_dir=checkpoint_dir)
                )

                self.assertEqual(resolved, str(checkpoint_dir.resolve()))
                self.assertEqual(
                    os.environ[PROMPT_CHECKPOINT_CACHE_DIR_ENV],
                    str(checkpoint_dir.resolve()),
                )
        finally:
            if old_value is None:
                os.environ.pop(PROMPT_CHECKPOINT_CACHE_DIR_ENV, None)
            else:
                os.environ[PROMPT_CHECKPOINT_CACHE_DIR_ENV] = old_value

    def test_setup_arg_parser_loop_guard_options(self):
        args = setup_arg_parser().parse_args(
            [
                "--loop-guard-ngram-size",
                "32",
                "--decode-progress-interval-tokens",
                "256",
                "--prefill-progress-interval-tokens",
                "2048",
                "--loop-guard-repeats",
                "4",
                "--loop-guard-min-tokens",
                "128",
            ]
        )

        self.assertEqual(args.loop_guard_ngram_size, 32)
        self.assertEqual(args.decode_progress_interval_tokens, 256)
        self.assertEqual(args.prefill_progress_interval_tokens, 2048)
        self.assertEqual(args.loop_guard_repeats, 4)
        self.assertEqual(args.loop_guard_min_tokens, 128)

    def test_glm_kv_bits_can_use_batch_generation_path(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        args = types.SimpleNamespace(seed=None)

        generator.model_provider = types.SimpleNamespace(
            is_batchable=True,
            cli_args=types.SimpleNamespace(kv_bits=None),
            model=object(),
        )
        self.assertTrue(generator._is_batchable(args))

        generator.model_provider = types.SimpleNamespace(
            is_batchable=True,
            cli_args=types.SimpleNamespace(kv_bits=8),
            model=object(),
        )
        self.assertFalse(generator._is_batchable(args))

        glm_model = types.SimpleNamespace(
            config={"model_type": "glm_moe_dsa", "indexer_types": ["full"]},
            layers=[
                types.SimpleNamespace(
                    self_attn=types.SimpleNamespace(skip_topk=False)
                )
            ],
        )
        generator.model_provider = types.SimpleNamespace(
            is_batchable=True,
            cli_args=types.SimpleNamespace(kv_bits=8),
            model=glm_model,
        )
        self.assertTrue(generator._is_batchable(args))

        generator.model_provider.cli_args.disable_batching = True
        self.assertFalse(generator._is_batchable(args))

    def test_single_request_passes_prompt_checkpoint_coexistence_args(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        cli_args = types.SimpleNamespace(
            prefill_step_size=2048,
            prefill_max_qk_tokens=67_108_864,
            checkpoint_min_tokens=DEFAULT_PROMPT_CHECKPOINT_MIN_TOKENS,
            checkpoint_cold_max_tokens=DEFAULT_PROMPT_CHECKPOINT_COLD_MAX_TOKENS,
            checkpoint_boundary_trim_tokens=(
                DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_TRIM_TOKENS
            ),
            checkpoint_boundary_align_tokens=(
                DEFAULT_PROMPT_CHECKPOINT_BOUNDARY_ALIGN_TOKENS
            ),
            checkpoint_continued_interval_tokens=(
                DEFAULT_PROMPT_CHECKPOINT_CONTINUED_INTERVAL_TOKENS
            ),
            checkpoint_save_exact="disabled",
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=4096,
        )
        tokenizer = types.SimpleNamespace(
            has_thinking=False,
            has_tool_calling=False,
            tool_parser=None,
            eos_token_id=0,
            encode=lambda text: [0],
        )
        generator.model_provider = types.SimpleNamespace(
            model=object(),
            tokenizer=tokenizer,
            draft_model=None,
            model_key=("model", None, None),
            cli_args=cli_args,
        )
        prompt = [1, 2, 3, 4, 5]
        rest = prompt[2:]

        class FakePromptCache:
            def fetch_nearest_cache(self, model_key, tokens):
                return ["server-cache"], rest

            def insert_cache(self, model_key, tokens, cache):
                self.inserted = (model_key, tokens, cache)

        class FakeStateMachine:
            def make_state(self):
                return "normal"

            def match(self, state, token):
                return state, None, "normal"

        generator.prompt_cache = FakePromptCache()
        generator._log_cache_stats = lambda: None
        def fake_tokenize(tokenizer, request, args, return_rendered=False):
            result = (
                prompt,
                [prompt[:3], prompt[3:]],
                ["system", "user"],
                "normal",
            )
            if return_rendered:
                return (*result, None)
            return result

        generator._tokenize = fake_tokenize
        generator._make_state_machine = lambda *args, **kwargs: (
            FakeStateMachine(),
            {},
        )
        request = object()
        args = types.SimpleNamespace(
            seed=None,
            stop_words=[],
            max_tokens=1,
            num_draft_tokens=3,
            top_logprobs=0,
            sampling=types.SimpleNamespace(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
            ),
            logits=types.SimpleNamespace(
                logit_bias=None,
                repetition_penalty=None,
                repetition_context_size=20,
                presence_penalty=0.0,
                presence_context_size=20,
                frequency_penalty=0.0,
                frequency_context_size=20,
            ),
        )
        captured = {}

        def fake_stream_generate(**kwargs):
            captured.update(kwargs)
            yield types.SimpleNamespace(
                finish_reason="length",
                token=0,
                text="",
                logprobs=mx.array([0.0]),
            )

        rqueue = Queue()
        with mock.patch("mlx_lm.server.stream_generate", fake_stream_generate):
            generator._serve_single((rqueue, request, args))

        queued = []
        while not rqueue.empty():
            queued.append(rqueue.get())
        self.assertFalse(any(isinstance(item, Exception) for item in queued))
        self.assertEqual(captured["prompt"], rest)
        self.assertEqual(captured["prompt_cache"], ["server-cache"])
        self.assertEqual(captured["prompt_checkpoint_full_prompt"], prompt)
        self.assertEqual(captured["prompt_checkpoint_initial_cached_tokens"], 2)
        self.assertEqual(captured["prompt_checkpoint_store_prefix_lengths"], [3])
        self.assertTrue(captured["prompt_checkpoint_allow_existing_cache"])
        self.assertFalse(captured["prompt_checkpoint_save_exact"])
        self.assertEqual(captured["prompt_checkpoint_frontier_min_tokens"], 10_240)
        self.assertEqual(captured["prompt_checkpoint_frontier_stride_tokens"], 10_240)
        self.assertEqual(captured["prefill_max_qk_tokens"], 67_108_864)
        self.assertEqual(captured["kv_bits"], 8)
        self.assertEqual(captured["kv_group_size"], 64)
        self.assertEqual(captured["quantized_kv_start"], 4096)

    def test_tokenize_segments_system_prefix_without_chat_template(self):
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = types.SimpleNamespace(
            cli_args=types.SimpleNamespace(chat_template_args={})
        )
        tokenizer = types.SimpleNamespace(
            has_chat_template=False,
            has_thinking=False,
            encode=lambda text: [ord(ch) for ch in text],
        )
        request = types.SimpleNamespace(
            request_type="chat",
            messages=[
                {"role": "system", "content": "stable AGENT prefix"},
                {"role": "user", "content": "changing task"},
            ],
            tools=None,
            role_mapping=None,
        )

        prompt, segments, segment_types, initial_state = generator._tokenize(
            tokenizer,
            request,
            types.SimpleNamespace(chat_template_kwargs=None),
        )

        self.assertEqual(initial_state, "normal")
        self.assertEqual(segment_types, ["system", "user"])
        self.assertEqual(segments[0] + segments[1], prompt)
        self.assertGreater(len(segments[0]), 0)
        self.assertLess(len(segments[0]), len(prompt))


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

        sm, _ = self.response_generator._make_state_machine(
            ("fake-empty-end", None, None),
            FakeTokenizer(),
            stop_words=[],
        )
        state = sm.make_state()
        state, _, s = sm.match(state, 100)
        self.assertEqual(s, "tool")
        for tok in [42, 43, 44]:
            state, _, s = sm.match(state, tok)
            self.assertEqual(s, "tool")
        state, _, s = sm.match(state, 2)
        self.assertIsNone(s)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        # Fetching a cache that is strictly a prefix doesn't remove it from the
        # lru cache
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 1)

        # Inserting a trimmable cache with shared prefix removes the prefixes
        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 1)

        # Fetching a cache with a shared prefix doesn't remove it either
        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache), 1)

        # Inserting a diverged cache actually creates another entry
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 2)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [1])
        c, t = cache.fetch_nearest_cache(model, [1, 3, 4])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [3, 4])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [2, 4, 5])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4, 5])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

    def test_insert_trimmable_cache_removes_immediate_prefix(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ab")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)

        cache.insert_cache(model, [1, 2, 3], [MockCache("abc")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 3)

    def test_insert_empty_tokens_does_not_self_destruct(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNotNone(c)
        self.assertEqual(t, [])

    def test_fetch_empty_tokens_after_root_eviction(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        cache.insert_cache(model, [1], [MockCache("a")])

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNone(c)
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])


if __name__ == "__main__":
    unittest.main()
