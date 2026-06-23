# Copyright © 2024 Apple Inc.

import copy
import json
import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.generate import (
    PROMPT_CHECKPOINT_DEBUG_ENV,
    generate_step,
    maybe_quantize_kv_cache,
    setup_arg_parser,
)
from mlx_lm.models.base import create_attention_mask, create_causal_mask
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    EMPTY_ARRAYS_METADATA_KEY,
    KVCache,
    PromptCacheCheckpointError,
    QuantizedGlmMlaKVCache,
    QuantizedKVCache,
    RotatingKVCache,
    expected_glm_mla_kv_quantization_metadata,
    expected_glm_mla_kv_settings_metadata,
    glm52_kv_cache_dir,
    glm52_local_cache_root,
    glm52_prompt_checkpoints_dir,
    load_prompt_checkpoint,
    load_prompt_cache,
    make_prompt_cache,
    prompt_checkpoint_file,
    prompt_prefix_hash,
    save_prompt_checkpoint,
    save_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.utils import load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestPromptCacheCheckpoint(unittest.TestCase):

    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_fid.name

    def tearDown(self):
        self.test_dir_fid.cleanup()

    def _set_home_to_test_dir(self):
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = self.test_dir

        def restore_home():
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home

        self.addCleanup(restore_home)

    def _set_prompt_checkpoint_debug(self):
        old_debug = os.environ.get(PROMPT_CHECKPOINT_DEBUG_ENV)
        os.environ[PROMPT_CHECKPOINT_DEBUG_ENV] = "1"

        def restore_debug():
            if old_debug is None:
                os.environ.pop(PROMPT_CHECKPOINT_DEBUG_ENV, None)
            else:
                os.environ[PROMPT_CHECKPOINT_DEBUG_ENV] = old_debug

        self.addCleanup(restore_debug)

    def _filled_kv_cache(self, shape=(1, 2, 4, 8), dtype=mx.float32):
        cache = [KVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=shape).astype(dtype)
            c.update_and_fetch(x, x)
        return cache

    def test_glm52_local_cache_paths(self):
        self._set_home_to_test_dir()
        expected_root = os.path.join(
            self.test_dir,
            ".cache",
            "mlx-lm",
            "glm52-local",
        )
        prompt = [1, 2, 3]

        self.assertEqual(glm52_local_cache_root(), expected_root)
        self.assertEqual(
            glm52_prompt_checkpoints_dir(),
            os.path.join(expected_root, "prompt-checkpoints"),
        )
        self.assertEqual(
            glm52_kv_cache_dir(),
            os.path.join(expected_root, "kv"),
        )
        self.assertEqual(
            prompt_checkpoint_file(prompt),
            os.path.join(
                expected_root,
                "prompt-checkpoints",
                f"{prompt_prefix_hash(prompt)}-3.safetensors",
            ),
        )

    def test_checkpoint_validates_prefix_namespace_and_model_hint(self):
        cache = self._filled_kv_cache()

        cache_file = os.path.join(self.test_dir, "checkpoint.safetensors")
        save_prompt_checkpoint(
            cache_file,
            cache,
            model_id="toy-model",
            prefix_tokens=[1, 2, 3, 4],
        )

        loaded_cache = load_prompt_checkpoint(
            cache_file,
            model_id="toy-model",
            prefix_tokens=[1, 2, 3, 4],
        )
        self.assertEqual(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                cache_file,
                model_id="toy-model",
                prefix_tokens=[1, 2, 3, 5],
            )
        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                cache_file,
                model_id="other-model",
                prefix_tokens=[1, 2, 3, 4],
            )
        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                cache_file,
                checkpoint_namespace="other-namespace",
                model_id="toy-model",
                prefix_tokens=[1, 2, 3, 4],
            )

    def test_checkpoint_rejects_cache_signature_mismatch(self):
        cache = self._filled_kv_cache()
        cache_file = os.path.join(self.test_dir, "checkpoint.safetensors")
        save_prompt_checkpoint(
            cache_file,
            cache,
            model_id="toy-model",
            prefix_tokens=[1, 2, 3, 4],
        )

        _, metadata = load_prompt_cache(cache_file, return_metadata=True)
        wrong_cache = self._filled_kv_cache(shape=(1, 2, 5, 8))
        bad_file = os.path.join(self.test_dir, "bad_signature.safetensors")
        save_prompt_cache(bad_file, wrong_cache, metadata)

        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                bad_file,
                model_id="toy-model",
                prefix_tokens=[1, 2, 3, 4],
            )

    def test_checkpoint_rejects_malformed_or_partial_metadata(self):
        cache = self._filled_kv_cache()
        cache_file = os.path.join(self.test_dir, "checkpoint.safetensors")
        save_prompt_checkpoint(
            cache_file,
            cache,
            model_id="toy-model",
            prefix_tokens=[1, 2, 3, 4],
        )
        _, metadata = load_prompt_cache(cache_file, return_metadata=True)

        malformed_cases = [
            ("checkpoint_prefix_length", "not-an-int"),
            ("checkpoint_glm_dsa_metadata", "{"),
        ]
        for key, value in malformed_cases:
            with self.subTest(key=key):
                bad_metadata = dict(metadata)
                bad_metadata[key] = value
                bad_file = os.path.join(self.test_dir, f"bad_{key}.safetensors")
                save_prompt_cache(bad_file, cache, bad_metadata)
                with self.assertRaises(PromptCacheCheckpointError):
                    load_prompt_checkpoint(
                        bad_file,
                        model_id="toy-model",
                        prefix_tokens=[1, 2, 3, 4],
                    )

        partial_metadata = dict(metadata)
        del partial_metadata["checkpoint_cache_signature"]
        partial_file = os.path.join(self.test_dir, "partial_metadata.safetensors")
        save_prompt_cache(partial_file, cache, partial_metadata)
        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                partial_file,
                model_id="toy-model",
                prefix_tokens=[1, 2, 3, 4],
            )

    def test_zero_sized_arrays_round_trip_shape_and_dtype(self):
        cache = [KVCache()]
        keys = mx.ones((1, 1, 3, 4), dtype=mx.float16)
        values = mx.zeros((1, 1, 3, 0), dtype=mx.float32)
        cache[0].update_and_fetch(keys, values)

        cache_file = os.path.join(self.test_dir, "zero_sized.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        loaded_keys, loaded_values = loaded_cache[0].state

        self.assertEqual(loaded_keys.shape, (1, 1, 3, 4))
        self.assertEqual(loaded_keys.dtype, mx.float16)
        self.assertEqual(loaded_values.shape, (1, 1, 3, 0))
        self.assertEqual(loaded_values.dtype, mx.float32)

    def test_empty_array_metadata_key_is_reserved(self):
        cache = self._filled_kv_cache()
        cache_file = os.path.join(self.test_dir, "reserved_key.safetensors")
        with self.assertRaises(ValueError):
            save_prompt_cache(
                cache_file,
                cache,
                {EMPTY_ARRAYS_METADATA_KEY: "{}"},
            )

    def _make_glm_moe_dsa_model(self, pattern="FSFS", kv_lora_rank=16):
        from mlx_lm.models import glm_moe_dsa

        args = glm_moe_dsa.ModelArgs(
            model_type="glm_moe_dsa",
            vocab_size=1024,
            hidden_size=128,
            index_head_dim=16,
            index_n_heads=4,
            index_topk=4,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=2.5,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=24,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=16,
            topk_method="noaux_tc",
            scoring_func="sigmoid",
            norm_topk_prob=True,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            moe_layer_freq=1,
            first_k_dense_replace=1,
            max_position_embeddings=1024,
            rms_norm_eps=1e-5,
            rope_parameters={"rope_theta": 10000.0},
            attention_bias=False,
            index_topk_pattern=pattern,
        )
        return glm_moe_dsa.Model(args)

    def _small_frontier_kwargs(self):
        return {
            "prefill_step_size": 5,
            "prompt_checkpoint_frontier_min_tokens": 4,
            "prompt_checkpoint_frontier_stride_tokens": 4,
        }

    def test_glm_moe_dsa_checkpoint_round_trip(self):
        model = self._make_glm_moe_dsa_model()
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        prefix_tokens = prompt[0].tolist()
        cache = make_prompt_cache(model)
        self.assertEqual([len(c.caches) for c in cache], [2, 1, 2, 1])

        logits = model(prompt, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)

        cache_file = os.path.join(self.test_dir, "glm_checkpoint.safetensors")
        save_prompt_checkpoint(
            cache_file,
            cache,
            model_id="glm-moe-dsa-test",
            prefix_tokens=prefix_tokens,
            model=model,
        )

        loaded_cache, metadata = load_prompt_checkpoint(
            cache_file,
            model_id="glm-moe-dsa-test",
            prefix_tokens=prefix_tokens,
            model=model,
            return_metadata=True,
        )
        self.assertEqual([len(c.caches) for c in loaded_cache], [2, 1, 2, 1])
        self.assertIn("checkpoint_glm_dsa_metadata", metadata)
        self.assertIn("checkpoint_glm_mla_kv_quantization", metadata)
        self.assertIn("checkpoint_glm_mla_kv_settings", metadata)
        self.assertEqual(
            json.loads(metadata["checkpoint_glm_mla_kv_settings"]),
            {
                "kv_bits": None,
                "kv_group_size": None,
                "quantized_kv_start": None,
            },
        )

        expected_int8 = expected_glm_mla_kv_quantization_metadata(
            model,
            cache_token_length=len(prefix_tokens),
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                cache_file,
                model_id="glm-moe-dsa-test",
                prefix_tokens=prefix_tokens,
                model=model,
                expected_glm_mla_kv_quantization=expected_int8,
                expected_glm_mla_kv_settings=expected_glm_mla_kv_settings_metadata(
                    model,
                    kv_bits=8,
                    kv_group_size=64,
                    quantized_kv_start=0,
                ),
            )

        live_logits = model(next_token, cache=cache)
        loaded_logits = model(next_token, cache=loaded_cache)
        mx.eval(live_logits, loaded_logits)
        self.assertTrue(mx.allclose(live_logits, loaded_logits).item())

        metadata["checkpoint_glm_dsa_metadata"] = json.dumps(
            {"model_type": "glm_moe_dsa", "indexer_types": ["full"]},
            sort_keys=True,
        )
        bad_file = os.path.join(self.test_dir, "bad_glm_checkpoint.safetensors")
        save_prompt_cache(bad_file, loaded_cache, metadata)
        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                bad_file,
                model_id="glm-moe-dsa-test",
                prefix_tokens=prefix_tokens,
                model=model,
            )

    def test_glm_moe_dsa_int8_checkpoint_round_trip_and_settings(self):
        model = self._make_glm_moe_dsa_model(kv_lora_rank=64)
        model.set_dtype(mx.float16)
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        prefix_tokens = prompt[0].tolist()
        cache = make_prompt_cache(model)

        logits = model(prompt, cache=cache)
        maybe_quantize_kv_cache(
            cache,
            quantized_kv_start=0,
            kv_group_size=64,
            kv_bits=8,
        )
        mx.eval(logits, [c.state for c in cache])
        for layer_cache in cache:
            self.assertIsInstance(layer_cache[0], QuantizedGlmMlaKVCache)
            if len(layer_cache.caches) > 1:
                self.assertIsInstance(layer_cache[1], KVCache)

        cache_file = os.path.join(self.test_dir, "glm_int8_checkpoint.safetensors")
        save_prompt_checkpoint(
            cache_file,
            cache,
            model_id="glm-moe-dsa-test",
            prefix_tokens=prefix_tokens,
            model=model,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
        )

        expected_quantization = expected_glm_mla_kv_quantization_metadata(
            model,
            cache_token_length=len(prefix_tokens),
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        expected_settings = expected_glm_mla_kv_settings_metadata(
            model,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        loaded_cache, metadata = load_prompt_checkpoint(
            cache_file,
            model_id="glm-moe-dsa-test",
            prefix_tokens=prefix_tokens,
            model=model,
            expected_glm_mla_kv_quantization=expected_quantization,
            expected_glm_mla_kv_settings=expected_settings,
            return_metadata=True,
        )
        self.assertEqual(
            json.loads(metadata["checkpoint_glm_mla_kv_settings"]),
            expected_settings,
        )
        for layer_cache in loaded_cache:
            self.assertIsInstance(layer_cache[0], QuantizedGlmMlaKVCache)
            self.assertEqual(layer_cache[0].group_size, 64)
            self.assertEqual(layer_cache[0].bits, 8)
            if len(layer_cache.caches) > 1:
                self.assertIsInstance(layer_cache[1], KVCache)

        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                cache_file,
                model_id="glm-moe-dsa-test",
                prefix_tokens=prefix_tokens,
                model=model,
                expected_glm_mla_kv_quantization=expected_quantization,
                expected_glm_mla_kv_settings=expected_glm_mla_kv_settings_metadata(
                    model,
                    kv_bits=8,
                    kv_group_size=32,
                    quantized_kv_start=0,
                ),
            )

        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                cache_file,
                model_id="glm-moe-dsa-test",
                prefix_tokens=prefix_tokens,
                model=model,
                expected_glm_mla_kv_quantization=expected_quantization,
                expected_glm_mla_kv_settings=expected_glm_mla_kv_settings_metadata(
                    model,
                    kv_bits=8,
                    kv_group_size=64,
                    quantized_kv_start=1,
                ),
            )

        with self.assertRaises(PromptCacheCheckpointError):
            load_prompt_checkpoint(
                cache_file,
                model_id="glm-moe-dsa-test",
                prefix_tokens=prefix_tokens,
                model=model,
                expected_glm_mla_kv_settings=expected_glm_mla_kv_settings_metadata(
                    model,
                    kv_bits=None,
                    kv_group_size=None,
                    quantized_kv_start=None,
                ),
            )

    def test_glm_moe_dsa_generation_prompt_checkpoint_smoke(self):
        self._set_home_to_test_dir()
        model = self._make_glm_moe_dsa_model()
        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        checkpoint_file = prompt_checkpoint_file(prompt.tolist())

        baseline = list(
            generate_step(
                prompt,
                model,
                max_tokens=3,
                prefill_step_size=3,
                prompt_checkpoint=False,
            )
        )
        miss_progress = []
        miss = list(
            generate_step(
                prompt,
                model,
                max_tokens=3,
                prefill_step_size=3,
                prompt_progress_callback=lambda processed, total: miss_progress.append(
                    (processed, total)
                ),
            )
        )
        self.assertTrue(os.path.exists(checkpoint_file))
        self.assertTrue(os.path.isdir(glm52_prompt_checkpoints_dir()))
        self.assertTrue(os.path.isdir(glm52_kv_cache_dir()))
        self.assertEqual(miss_progress[0], (0, len(prompt)))

        progress = []
        hit = list(
            generate_step(
                prompt,
                model,
                max_tokens=3,
                prefill_step_size=3,
                prompt_progress_callback=lambda processed, total: progress.append(
                    (processed, total)
                ),
            )
        )

        self.assertEqual(progress[0], (len(prompt) - 1, len(prompt)))
        self.assertEqual([tok for tok, _ in miss], [tok for tok, _ in baseline])
        self.assertEqual([tok for tok, _ in hit], [tok for tok, _ in baseline])
        for (_, expected), (_, from_hit) in zip(baseline, hit):
            self.assertTrue(mx.allclose(expected, from_hit).item())

        backup_root = glm52_local_cache_root() + ".bak"
        os.rename(glm52_local_cache_root(), backup_root)
        self.assertFalse(os.path.exists(glm52_local_cache_root()))

        recreated_progress = []
        recreated = list(
            generate_step(
                prompt,
                model,
                max_tokens=3,
                prefill_step_size=3,
                prompt_progress_callback=lambda processed, total: recreated_progress.append(
                    (processed, total)
                ),
            )
        )
        self.assertEqual(recreated_progress[0], (0, len(prompt)))
        self.assertTrue(os.path.exists(checkpoint_file))
        self.assertEqual([tok for tok, _ in recreated], [tok for tok, _ in baseline])

    def test_prompt_checkpoint_reuses_stable_prefix_with_different_suffix(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        model = self._make_glm_moe_dsa_model()
        stable_prefix = [1, 2, 3, 4]
        prompt_a = mx.array(stable_prefix + [10, 11, 12])
        prompt_b = mx.array(stable_prefix + [20, 21, 22])
        prefix_file = prompt_checkpoint_file(stable_prefix)

        baseline_b = list(
            generate_step(
                prompt_b,
                model,
                max_tokens=2,
                prefill_step_size=2,
                prompt_checkpoint=False,
            )
        )

        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(
                generate_step(
                    prompt_a,
                    model,
                    max_tokens=2,
                    prefill_step_size=2,
                    prompt_checkpoint_store_prefix_lengths=[len(stable_prefix)],
                )
            )
        first_run = "\n".join(logs.output)
        self.assertTrue(os.path.exists(prefix_file))
        self.assertIn("save prefix success", first_run)

        progress = []
        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            hit_b = list(
                generate_step(
                    prompt_b,
                    model,
                    max_tokens=2,
                    prefill_step_size=2,
                    prompt_progress_callback=lambda processed, total: progress.append(
                        (processed, total)
                    ),
                    prompt_checkpoint_store_prefix_lengths=[len(stable_prefix)],
                )
            )
        second_run = "\n".join(logs.output)
        self.assertIn("prompt checkpoint: prefix hit", second_run)
        self.assertIn(f"prefix_length={len(stable_prefix)}", second_run)
        self.assertEqual(progress[0], (len(stable_prefix), len(prompt_b)))
        self.assertEqual([tok for tok, _ in hit_b], [tok for tok, _ in baseline_b])
        for (_, expected), (_, from_hit) in zip(baseline_b, hit_b):
            self.assertTrue(mx.allclose(expected, from_hit).item())

    def test_long_prompt_saves_multiple_frontier_checkpoints(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        model = self._make_glm_moe_dsa_model()
        prompt_tokens = list(range(1, 16))

        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(
                generate_step(
                    mx.array(prompt_tokens),
                    model,
                    max_tokens=1,
                    **self._small_frontier_kwargs(),
                )
            )

        output = "\n".join(logs.output)
        for prefix_length in (4, 8, 12):
            self.assertTrue(
                os.path.exists(prompt_checkpoint_file(prompt_tokens[:prefix_length]))
            )
        self.assertTrue(os.path.exists(prompt_checkpoint_file(prompt_tokens)))
        self.assertEqual(output.count("save frontier success"), 3)
        self.assertIn("fresh_prefill_tokens=14", output)
        self.assertIn("prefill chunk", output)

    def test_prompt_checkpoint_reuses_deepest_frontier_after_restart(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        model = self._make_glm_moe_dsa_model()
        common = list(range(1, 13))
        prompt_a = common + [101, 102, 103]
        prompt_b = common + [201, 202, 203]

        baseline_b = list(
            generate_step(
                mx.array(prompt_b),
                model,
                max_tokens=1,
                prompt_checkpoint=False,
                prefill_step_size=5,
            )
        )
        with self.assertLogs("mlx_lm.generate", level="INFO"):
            list(
                generate_step(
                    mx.array(prompt_a),
                    model,
                    max_tokens=1,
                    **self._small_frontier_kwargs(),
                )
            )

        progress = []
        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            hit_b = list(
                generate_step(
                    mx.array(prompt_b),
                    model,
                    max_tokens=1,
                    prompt_progress_callback=lambda processed, total: progress.append(
                        (processed, total)
                    ),
                    **self._small_frontier_kwargs(),
                )
            )

        output = "\n".join(logs.output)
        self.assertIn("prompt checkpoint: frontier hit", output)
        self.assertIn("prefix_length=12", output)
        self.assertIn("disk_cached_tokens=12", output)
        self.assertIn("fresh_prefill_tokens=2", output)
        self.assertEqual(progress[0], (12, len(prompt_b)))
        self.assertEqual([tok for tok, _ in hit_b], [tok for tok, _ in baseline_b])
        for (_, expected), (_, from_hit) in zip(baseline_b, hit_b):
            self.assertTrue(mx.allclose(expected, from_hit).item())

    def test_server_prompt_cache_coverage_wins_over_disk_frontier(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        model = self._make_glm_moe_dsa_model()
        common = list(range(1, 13))
        prompt_a = common + [101, 102, 103]
        prompt_b = common + [201, 202, 203]

        with self.assertLogs("mlx_lm.generate", level="INFO"):
            list(
                generate_step(
                    mx.array(prompt_a),
                    model,
                    max_tokens=1,
                    **self._small_frontier_kwargs(),
                )
            )

        server_cache = make_prompt_cache(model)
        logits = model(mx.array([prompt_b[:13]]), cache=server_cache)
        mx.eval(logits, [c.state for c in server_cache])

        progress = []
        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(
                generate_step(
                    mx.array(prompt_b[13:]),
                    model,
                    max_tokens=1,
                    prompt_cache=server_cache,
                    prompt_checkpoint_full_prompt=prompt_b,
                    prompt_checkpoint_initial_cached_tokens=13,
                    prompt_checkpoint_allow_existing_cache=True,
                    prompt_progress_callback=lambda processed, total: progress.append(
                        (processed, total)
                    ),
                    **self._small_frontier_kwargs(),
                )
            )

        output = "\n".join(logs.output)
        self.assertIn("server prompt_cache coexistence", output)
        self.assertIn("miss covered by server prompt_cache", output)
        self.assertIn("resolution=server-cache-covered", output)
        self.assertNotIn("prompt checkpoint: frontier hit", output)
        self.assertEqual(progress[0], (13, len(prompt_b)))

    def test_malformed_longest_frontier_falls_back_to_shorter_valid_frontier(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        model = self._make_glm_moe_dsa_model()
        common = list(range(1, 13))
        prompt_a = common + [101, 102, 103]
        prompt_b = common + [201, 202, 203]

        with self.assertLogs("mlx_lm.generate", level="INFO"):
            list(
                generate_step(
                    mx.array(prompt_a),
                    model,
                    max_tokens=1,
                    **self._small_frontier_kwargs(),
                )
            )
        with open(prompt_checkpoint_file(common), "wb") as f:
            f.write(b"not a safetensors checkpoint")

        progress = []
        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(
                generate_step(
                    mx.array(prompt_b),
                    model,
                    max_tokens=1,
                    prompt_progress_callback=lambda processed, total: progress.append(
                        (processed, total)
                    ),
                    **self._small_frontier_kwargs(),
                )
            )

        output = "\n".join(logs.output)
        self.assertIn("miss rejected", output)
        self.assertIn("prompt checkpoint: frontier hit", output)
        self.assertIn("prefix_length=8", output)
        self.assertEqual(progress[0], (8, len(prompt_b)))

    def test_prompt_checkpoint_coexists_with_server_managed_prompt_cache(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        model = self._make_glm_moe_dsa_model()
        stable_prefix = [1, 2, 3, 4]
        prompt_a = mx.array(stable_prefix + [10, 11, 12])
        prompt_b = stable_prefix + [20, 21, 22]

        with self.assertLogs("mlx_lm.generate", level="INFO"):
            list(
                generate_step(
                    prompt_a,
                    model,
                    max_tokens=1,
                    prefill_step_size=2,
                    prompt_checkpoint_store_prefix_lengths=[len(stable_prefix)],
                )
            )

        server_cache = make_prompt_cache(model)
        logits = model(mx.array([prompt_b[:2]]), cache=server_cache)
        mx.eval(logits, [c.state for c in server_cache])

        progress = []
        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(
                generate_step(
                    mx.array(prompt_b[2:]),
                    model,
                    max_tokens=1,
                    prefill_step_size=2,
                    prompt_cache=server_cache,
                    prompt_checkpoint_full_prompt=prompt_b,
                    prompt_checkpoint_initial_cached_tokens=2,
                    prompt_checkpoint_allow_existing_cache=True,
                    prompt_progress_callback=lambda processed, total: progress.append(
                        (processed, total)
                    ),
                )
            )
        output = "\n".join(logs.output)
        self.assertIn("server prompt_cache coexistence", output)
        self.assertNotIn("skip explicit prompt_cache active", output)
        self.assertIn("prompt checkpoint: prefix hit", output)
        self.assertEqual(progress[0], (len(stable_prefix), len(prompt_b)))

    def test_empty_prompt_with_explicit_prompt_cache_rejects_checkpoint_kwargs(self):
        model = self._make_glm_moe_dsa_model()
        prompt_cache = make_prompt_cache(model)

        with self.assertRaisesRegex(
            ValueError,
            "Either input_embeddings or prompt",
        ):
            list(
                generate_step(
                    mx.array([], dtype=mx.int32),
                    model,
                    max_tokens=1,
                    prompt_cache=prompt_cache,
                    prompt_checkpoint_full_prompt=[1, 2, 3, 4],
                )
            )

    def test_empty_prompt_with_server_managed_cache_replays_last_token(self):
        model = self._make_glm_moe_dsa_model()
        prompt_tokens = [1, 2, 3, 4]

        baseline = list(
            generate_step(
                mx.array(prompt_tokens),
                model,
                max_tokens=1,
                prompt_checkpoint=False,
            )
        )

        prompt_cache = make_prompt_cache(model)
        logits = model(mx.array([prompt_tokens]), cache=prompt_cache)
        mx.eval(logits, [c.state for c in prompt_cache])

        progress = []
        resumed = list(
            generate_step(
                mx.array([], dtype=mx.int32),
                model,
                max_tokens=1,
                prompt_cache=prompt_cache,
                prompt_checkpoint=False,
                prompt_checkpoint_full_prompt=prompt_tokens,
                prompt_checkpoint_initial_cached_tokens=len(prompt_tokens),
                prompt_checkpoint_allow_existing_cache=True,
                prompt_progress_callback=lambda processed, total: progress.append(
                    (processed, total)
                ),
            )
        )

        self.assertEqual(progress[0], (len(prompt_tokens) - 1, len(prompt_tokens)))
        self.assertEqual([tok for tok, _ in resumed], [tok for tok, _ in baseline])
        for (_, expected), (_, actual) in zip(baseline, resumed):
            self.assertTrue(mx.allclose(expected, actual).item())

    def test_prompt_checkpoint_debug_logging(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        model = self._make_glm_moe_dsa_model()
        prompt = mx.array([1, 2, 3, 4])

        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(generate_step(prompt, model, max_tokens=1, prefill_step_size=2))
        first_run = "\n".join(logs.output)
        self.assertIn("prompt checkpoint: lookup", first_run)
        self.assertIn("prefix_length=4", first_run)
        self.assertIn("miss file does not exist", first_run)
        self.assertIn("save exact success", first_run)

        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(generate_step(prompt, model, max_tokens=1, prefill_step_size=2))
        second_run = "\n".join(logs.output)
        self.assertIn("prompt checkpoint: lookup", second_run)
        self.assertIn("prompt checkpoint: exact hit", second_run)
        self.assertIn("prefix_length=4", second_run)

        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            list(generate_step(prompt, model, max_tokens=1, prompt_checkpoint=False))
        disabled = "\n".join(logs.output)
        self.assertIn("prompt checkpoint: checkpoint disabled", disabled)

    def test_prompt_checkpoint_can_be_disabled(self):
        self._set_home_to_test_dir()
        model = self._make_glm_moe_dsa_model()
        prompt = mx.array([1, 2, 3, 4])

        args = setup_arg_parser().parse_args(["--no-prompt-checkpoint"])
        self.assertTrue(args.no_prompt_checkpoint)

        list(generate_step(prompt, model, max_tokens=1, prompt_checkpoint=False))
        self.assertFalse(os.path.exists(glm52_local_cache_root()))

    def test_prompt_checkpoint_save_failure_does_not_stop_generation(self):
        self._set_home_to_test_dir()
        self._set_prompt_checkpoint_debug()
        os.makedirs(os.path.join(self.test_dir, ".cache"))
        with open(os.path.join(self.test_dir, ".cache", "mlx-lm"), "w"):
            pass

        model = self._make_glm_moe_dsa_model()
        prompt = mx.array([1, 2, 3, 4])
        with self.assertLogs("mlx_lm.generate", level="INFO") as logs:
            outputs = list(generate_step(prompt, model, max_tokens=1))

        self.assertEqual(len(outputs), 1)
        self.assertFalse(os.path.exists(glm52_local_cache_root()))
        self.assertIn("save exact failure swallowed", "\n".join(logs.output))


class TestPromptCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_load(self):
        cache = [KVCache() for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_save_load_rotating_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        # Test with rotating cache
        cache = [RotatingKVCache(max_size=8, keep=2) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.keep, lc.keep)
            self.assertEqual(c.max_size, lc.max_size)
            self.assertEqual(c.step, lc.step)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Do a couple single token updates to get a rotation
        for _ in range(2):
            for c in cache:
                x = mx.random.uniform(shape=(1, 8, 1, 4))
                c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        for c, lc in zip(cache, loaded_cache):
            x = mx.random.uniform(shape=(1, 8, 1, 4))
            k, v = c.update_and_fetch(x, x)
            lk, lv = lc.update_and_fetch(x, x)
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(k, lk))
            self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_mixed_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache, loaded_cache):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_cache_list(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)
        cache = [CacheList(*cache)]

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache[0].caches, loaded_cache[0].caches):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_arrays_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [ArraysCache(size=2)]
        cache[0][0] = mx.zeros((1, 4, 4))
        cache[0][1] = mx.zeros((1, 4, 4))

        save_prompt_cache(cache_file, cache)
        loaded = load_prompt_cache(cache_file)

        # Try to make a mask
        mask = loaded[0].make_mask(4)

    def test_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = list(generate_step(prompt, model, max_tokens=4))
        toks, all_logits = zip(*results)

        prompt_cache = make_prompt_cache(model)
        i = 0
        for tok, logits in generate_step(
            prompt, model, prompt_cache=prompt_cache, max_tokens=2
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        for tok, logits in generate_step(
            mx.array([toks[i]]), model, prompt_cache=prompt_cache, max_tokens=1
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))

    def test_trim_cache(self):
        cache = [KVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        # Trim
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

        # Can't trim arrays cache
        cache = [ArraysCache(size=2) for _ in range(2)]
        for c in cache:
            c[0] = mx.zeros((5, 5))
            c[1] = mx.zeros((5, 5))
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 0)

        # All cache's have to be trimmable
        cache = [ArraysCache(size=2), KVCache()]
        cache[0][0] = mx.zeros((5, 5))
        cache[0][1] = mx.zeros((5, 5))
        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[1].update_and_fetch(x, x)
        num_trimmed = trim_prompt_cache(cache, 1)
        self.assertEqual(num_trimmed, 0)

        cache = [RotatingKVCache(max_size=6) for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 5, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 4)

        # Can't trim fixed-size KV cache after processing
        # more than max_kv_size tokens
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 0)

        cache = [QuantizedKVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 64))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

    def test_trim_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]

        prompt_cache = make_prompt_cache(model)

        # Generate one token so we process the full prompt
        last_tok, _ = next(generate_step(prompt, model, prompt_cache=prompt_cache))
        last_tok = mx.array([last_tok])

        # Generate two more tokens
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        toks, all_logits = zip(*(r[1] for r in results))

        # To get back to the cache just after processing the prompt,
        # trim by 3 tokens
        trim_prompt_cache(prompt_cache, 3)

        # Generate the same thing again
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        second_toks, second_all_logits = zip(*(r[1] for r in results))
        self.assertEqual(toks, second_toks)
        self.assertTrue(
            all(mx.allclose(l, l2) for l, l2 in zip(all_logits, second_all_logits))
        )

    def test_cache_copying(self):
        cache = [KVCache()]

        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[0].update_and_fetch(x, x)

        y = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(y, y)

        old_cache = copy.deepcopy(cache)

        trim_prompt_cache(cache, 1)

        self.assertTrue(old_cache[0].offset, 11)
        self.assertTrue(cache[0].offset, 10)

        z = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(z, z)

        self.assertTrue(mx.allclose(old_cache[0].keys[..., 10:11, :], y))
        self.assertTrue(mx.allclose(cache[0].keys[..., 10:11, :], z))

    def test_save_load_quantized_cache(self):
        cache = [QuantizedKVCache(bits=4, group_size=32) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 32))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(loaded_cache[0].bits == cache[0].bits)
        self.assertTrue(loaded_cache[0].group_size == cache[0].group_size)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            # Loop over quantized tuple
            for i in range(3):
                self.assertTrue(mx.array_equal(c.state[0][i], lc.state[0][i]))
                self.assertTrue(mx.array_equal(c.state[1][i], lc.state[1][i]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_cache_to_quantized(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = zip(range(4), generate_step(prompt, model))
        toks, all_logits = zip(*(r[1] for r in results))

        prompt_cache = make_prompt_cache(model)
        i = 0
        for _, (tok, logits) in zip(
            range(2), generate_step(prompt, model, prompt_cache=prompt_cache)
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        prompt_cache = [c.to_quantized(bits=8, group_size=32) for c in prompt_cache]

        for _, (tok, logits) in zip(
            range(1),
            generate_step(mx.array([toks[i]]), model, prompt_cache=prompt_cache),
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i], rtol=4e-2))

    def test_cache_list(self):
        c = CacheList(KVCache(), KVCache())
        self.assertTrue(c.is_trimmable())
        k = mx.zeros((1, 2, 8, 8))
        v = mx.zeros((1, 2, 8, 8))
        c[0].update_and_fetch(k, v)
        c[1].update_and_fetch(k, v)
        m = c.trim(5)
        self.assertEqual(m, 5)

        c = CacheList(ArraysCache(size=2), KVCache())
        self.assertFalse(c.is_trimmable())

        c1 = CacheList(ArraysCache(size=1), KVCache())
        c1[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c1[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 5, 4)), mx.random.normal(shape=(1, 2, 5, 4))
        )

        c2 = CacheList(ArraysCache(size=1), KVCache())
        c2[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c2[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 7, 4)), mx.random.normal(shape=(1, 2, 7, 4))
        )

        merged_cache = CacheList.merge((c1, c2))
        c1_ex = merged_cache.extract(0)
        self.assertTrue(mx.array_equal(c1_ex[0][0], c1[0][0]))
        self.assertTrue(mx.array_equal(c1_ex[1].state[0], c1[1].state[0]))
        c2_ex = merged_cache.extract(1)
        self.assertTrue(mx.array_equal(c2_ex[0][0], c2[0][0]))
        self.assertTrue(mx.array_equal(c2_ex[1].state[0], c2[1].state[0]))

    def test_make_mask_with_cache(self):
        # For 1 time step with no cache, don't need a mask
        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=False)
        self.assertEqual(mask, None)

        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=True)
        self.assertEqual(mask, None)

        # Regular causal mask
        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        # With a window size
        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=4, return_array=False
        )
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=3, return_array=False
        )
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4, window_size=3)))

        # With a regular KV cache
        cache = KVCache()
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        k = v = mx.zeros((1, 2, 16, 8))
        cache.update_and_fetch(k, v)
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertEqual(mask.shape, (4, 20))

    def test_rotating_cache_mask(self):
        cache = RotatingKVCache(max_size=8)

        mask = cache.make_mask(4, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(mx.zeros((1, 4, 32)), cache, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(
            mx.zeros((1, 4, 32)), cache, window_size=5, return_array=True
        )
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.shape, (4, 4))

        mask = cache.make_mask(6, window_size=5)
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.sum(axis=-1).max(), 5)
        cmask = create_attention_mask(mx.zeros((1, 6, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertEqual(mask, None)

        kv = mx.zeros((1, 1, 10, 32))
        cache.update_and_fetch(kv, kv)
        mask = cache.make_mask(3, window_size=5)
        self.assertEqual(mask.shape, (3, 10))
        self.assertTrue(mx.all(mask.sum(axis=-1) == 5))
        for i in range(3):
            s = 11 - 3 + i
            self.assertTrue(mx.all(mask[s - 5 : s]))
        cmask = create_attention_mask(mx.zeros((1, 3, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache)
        self.assertEqual(mask, None)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] + [False] * 3 + [True] * 4)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        kv = mx.zeros((1, 1, 1, 32))
        cache.update_and_fetch(kv, kv)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] * 2 + [False] * 3 + [True] * 3)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

    def test_batch_kv_cache(self):
        cache = BatchKVCache(left_padding=[2, 3, 4])
        k, v = mx.zeros((3, 1, 4, 8)), mx.zeros((3, 1, 4, 8))
        # Update works
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape, (3, 1, 4, 8))

        # State can be evaluated
        mx.eval(cache.state)

        # State can be set
        cache.state = cache.state

        # Test filtering
        cache.filter([0, 1])

        # In this case filtering left shifts the cache so it has zero padding
        self.assertEqual(cache.state[0].shape, (2, 1, 2, 8))

        mask = cache.make_mask(1)
        self.assertEqual(mask[0].squeeze().tolist(), [True, True, True])
        self.assertEqual(mask[1].squeeze().tolist(), [False, True, True])

        # Test extension
        cache_a = BatchKVCache(left_padding=[2, 1, 2])
        cache_b = BatchKVCache(left_padding=[3, 0])

        k = mx.zeros((3, 1, 8, 1))
        v = mx.zeros((3, 1, 8, 1))
        cache_a.update_and_fetch(k, v)

        k = mx.zeros((2, 1, 4, 1))
        v = mx.zeros((2, 1, 4, 1))
        cache_b.update_and_fetch(k, v)

        cache_a.extend(cache_b)
        self.assertEqual(cache_a.keys.shape[0], 5)
        self.assertEqual(cache_a.values.shape[0], 5)
        self.assertEqual(cache_a.offset.tolist(), [6, 7, 6, 1, 4])
        self.assertEqual(cache_a.left_padding.tolist(), [2, 1, 2, 7, 4])

    def test_batch_rotating_kv_cache(self):
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        mask = cache.make_mask(4)
        self.assertFalse(mx.any(mask[0, 0, 0, :]))
        self.assertTrue(
            mx.array_equal(mask[1, 0, 0, :], mx.array([True, False, False, False]))
        )

        # Batch update works
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(4)
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (4, k.shape[2]))
        self.assertEqual(
            mask[0, 0, 0, :].tolist(), [False, True, True, True, False, False, False]
        )

        # Single query update works
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))

        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (1, k.shape[2]))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, False, True, True])
        self.assertEqual(mask[1, 0, 0].tolist(), [True, True, True, True])

        # Check filtering
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0, 3])
        k, v = mx.zeros((3, 1, 3, 8)), mx.zeros((3, 1, 3, 8))
        cache.update_and_fetch(k, v)
        cache.filter(mx.array([1]))
        self.assertEqual(cache.keys.shape, (1, 1, 3, 8))

        # Check extend
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 1])
        other = BatchRotatingKVCache(max_size=4, left_padding=[2, 2])
        k, v = mx.zeros((2, 1, 5, 8)), mx.zeros((2, 1, 5, 8))
        cache.update_and_fetch(k, v)
        other.update_and_fetch(k, v)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))
        cache.update_and_fetch(k, v)
        cache.extend(other)

        # Check mask when going from prompt -> extend -> prompt
        cache = BatchRotatingKVCache(max_size=8, left_padding=[4])
        k, v = mx.zeros((1, 1, 8, 8)), mx.zeros((1, 1, 8, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        self.assertEqual(
            mask.squeeze().tolist(), [True, False, False, False, True, True, True, True]
        )

        k, v = mx.zeros((1, 1, 1, 8)), mx.zeros((1, 1, 1, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(2)
        expected = mx.array(
            [
                [False, False, False, True, True, True, True, True, False],
                [False, False, False, True, True, True, True, True, True],
            ]
        )
        self.assertTrue(mx.array_equal(mask.squeeze(), expected))

    def test_save_load_batch_caches(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2, left_padding=[1, 2]),
            BatchKVCache(left_padding=[1, 2]),
            BatchRotatingKVCache(max_size=10, left_padding=[1, 2]),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        left_padding = mx.array([1, 2])
        for c, lc in zip(cache, loaded_cache):
            self.assertTrue(mx.array_equal(c.left_padding, left_padding))

    def test_rotating_cache_updates(self):
        cache = RotatingKVCache(max_size=8)
        k = v = mx.zeros((1, 1, 10, 1))
        cache.update_and_fetch(k, v)

        for _ in range(3):
            k = v = mx.zeros((1, 1, 1, 1))
            cache.update_and_fetch(k, v)

        k = v = mx.zeros((1, 1, 3, 1))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape[2], 10)
        self.assertEqual(v.shape[2], 10)

    def test_merge_with_empty_caches(self):
        c1 = ArraysCache(2)
        c2 = ArraysCache(2)
        c2[0] = mx.zeros((1, 4))
        c2[1] = mx.zeros((1, 4))
        c_out = ArraysCache.merge((c1, c2))
        self.assertEqual(c_out[0].shape, (2, 4))
        self.assertEqual(c_out[1].shape, (2, 4))

        c1 = KVCache()
        c2 = KVCache()
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

        c1 = RotatingKVCache(max_size=4)
        c2 = RotatingKVCache(max_size=4)
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

    def test_extend_with_empty_and_nonempty_batch_caches(self):
        """Extending a batch cache when one side has keys=None should use the
        correct batch size for the placeholder, not the batch size from the
        non-None side. Regression test for broadcast error in dynamic_roll."""
        H, D = 8, 64
        max_size = 512

        # -- BatchRotatingKVCache --
        # Create 2 caches with content and 3 empty caches
        c1 = RotatingKVCache(max_size=max_size)
        c2 = RotatingKVCache(max_size=max_size)
        c1.update_and_fetch(mx.ones((1, H, 5, D)), mx.ones((1, H, 5, D)))
        c2.update_and_fetch(mx.ones((1, H, 3, D)), mx.ones((1, H, 3, D)))
        batch_full = BatchRotatingKVCache.merge([c1, c2])

        empty_caches = [RotatingKVCache(max_size=max_size) for _ in range(3)]
        batch_empty = BatchRotatingKVCache.merge(empty_caches)

        # Extend non-empty with empty (different batch sizes)
        batch_full.extend(batch_empty)
        self.assertEqual(batch_full.keys.shape[0], 5)
        self.assertEqual(batch_full.offset.shape[0], 5)

        # Prompt processing with right padding should not crash
        batch_full.prepare(lengths=[10, 8, 12, 7, 11], right_padding=[2, 4, 0, 5, 1])
        new_kv = mx.ones((5, H, 12, D))
        batch_full.update_and_fetch(new_kv, new_kv)

        # Also test empty extending non-empty
        batch_full2 = BatchRotatingKVCache.merge(
            [RotatingKVCache(max_size=max_size) for _ in range(3)]
        )
        c3 = RotatingKVCache(max_size=max_size)
        c4 = RotatingKVCache(max_size=max_size)
        c3.update_and_fetch(mx.ones((1, H, 4, D)), mx.ones((1, H, 4, D)))
        c4.update_and_fetch(mx.ones((1, H, 6, D)), mx.ones((1, H, 6, D)))
        batch_content = BatchRotatingKVCache.merge([c3, c4])
        batch_full2.extend(batch_content)
        self.assertEqual(batch_full2.keys.shape[0], 5)
        self.assertEqual(batch_full2.offset.shape[0], 5)

        # -- BatchKVCache --
        c1 = KVCache()
        c2 = KVCache()
        c1.update_and_fetch(mx.ones((1, H, 5, D)), mx.ones((1, H, 5, D)))
        c2.update_and_fetch(mx.ones((1, H, 3, D)), mx.ones((1, H, 3, D)))
        batch_full = BatchKVCache.merge([c1, c2])

        empty_caches = [KVCache() for _ in range(3)]
        batch_empty = BatchKVCache.merge(empty_caches)

        batch_full.extend(batch_empty)
        self.assertEqual(batch_full.keys.shape[0], 5)
        self.assertEqual(batch_full.offset.shape[0], 5)

    def test_arrays_cache_extend_with_empty(self):
        # test simple merge
        c1 = ArraysCache(2)
        c2 = ArraysCache(2)
        c1[0] = mx.zeros((1, 4, 8))
        c1[1] = mx.zeros((1, 4))
        c2[0] = mx.zeros((1, 4, 8))
        c2[1] = mx.zeros((1, 4))
        full = ArraysCache.merge((c1, c2))
        self.assertEqual(full[0].shape, (2, 4, 8))

        # extend with empty
        empty = ArraysCache.merge((ArraysCache(2),))
        full.extend(empty)
        self.assertEqual(full[0].shape, (3, 4, 8))
        self.assertEqual(full[1].shape, (3, 4))
        self.assertTrue(mx.all(full[0][2:] == 0))

        # making an empty cache with 2 sequences and merging it with
        # another one with 2 sequences
        empty2 = ArraysCache.merge((ArraysCache(2), ArraysCache(2)))
        content = ArraysCache.merge((c1, c2))
        empty2.extend(content)
        self.assertEqual(empty2[0].shape, (4, 4, 8))
        self.assertEqual(empty2[1].shape, (4, 4))

        # Extend content with empty
        content = ArraysCache.merge((c1, c2))
        empty2 = ArraysCache.merge((ArraysCache(2), ArraysCache(2)))
        content.extend(empty2)
        self.assertEqual(content[0].shape, (4, 4, 8))
        self.assertEqual(content[1].shape, (4, 4))
        self.assertEqual(content.make_mask(10).shape, (4, 10))

        # multiple empty extensions accumulate correctly
        stepwise = ArraysCache.merge((c1,))
        stepwise.extend(ArraysCache(2))
        stepwise.extend(ArraysCache.merge((ArraysCache(2), ArraysCache(2))))
        self.assertEqual(stepwise[0].shape, (4, 4, 8))
        self.assertEqual(stepwise[1].shape, (4, 4))

    def test_window_mask_with_full_kv_cache(self):
        c = KVCache()
        kv = mx.zeros((1, 1, 32, 128))
        c.update_and_fetch(kv, kv)

        h = mx.zeros((1, 1, 1, 128))
        mask = create_attention_mask(h, c, window_size=4)
        expected = create_causal_mask(1, offset=32, window_size=4)
        self.assertTrue(mx.array_equal(mask, expected))


if __name__ == "__main__":
    unittest.main()
