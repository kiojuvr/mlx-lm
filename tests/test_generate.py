# Copyright © 2024 Apple Inc.

import importlib
import random
import unittest
from typing import List

import mlx.core as mx

generate_module = importlib.import_module("mlx_lm.generate")
from mlx_lm.generate import (
    BatchGenerator,
    GenerationResponse,
    SequenceStateMachine,
    _effective_prefill_step_size,
    _glm_dsa_adaptive_prefill_step_size,
    _glm_dsa_decode_profile_fields,
    _format_prefill_chunk_fields,
    _glm_dsa_prefill_profile_chunk_fields,
    batch_generate,
    generate,
    generate_step,
    mtp_speculative_generate_step,
    stream_generate,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load


class TestGenerateUtilities(unittest.TestCase):
    def test_mtp_speculative_shifts_prefill_and_reuses_first_draft(self):
        class DummyCache:
            def __init__(self):
                self.offset = 0

            @property
            def state(self):
                return mx.array([self.offset])

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.offset = max(0, self.offset - n)
                return n

        class DummyModel:
            def __init__(self):
                self.mtp = object()
                self.layers = [object()]
                self.args = type(
                    "Args", (), {"index_share_for_mtp_iteration": True}
                )()
                self.target_cache = [DummyCache()]
                self.mtp_cache = DummyCache()
                self.mtp_prefill_inputs = []
                self.mtp_last_prefill_inputs = []
                self.mtp_logits_inputs = []
                self.mtp_logits_prev_topk = []
                self.mtp_prefill_prev_topk = []
                self.target_inputs = []
                self.eval_phase = None

            def make_mtp_cache(self):
                return self.mtp_cache

            def _logits(self, tokens):
                rows = []
                for token in tokens:
                    logits = mx.where(
                        mx.arange(32) == token,
                        mx.array(1.0),
                        mx.array(0.0),
                    )
                    rows.append(logits.reshape(1, 1, 32))
                return mx.concatenate(rows, axis=1)

            def prefill_with_hidden(self, inputs, cache=None):
                self.eval_phase = "target"
                flat = [int(token) for token in inputs.reshape(-1).tolist()]
                self.target_inputs.append(flat)
                cache[0].offset += len(flat)
                hidden = mx.broadcast_to(
                    mx.array(flat, dtype=mx.float32).reshape(1, len(flat), 1),
                    (1, len(flat), 4),
                )
                return self._logits([flat[-1] + 1]), hidden

            def forward_with_hidden(self, inputs, cache=None):
                self.eval_phase = "target"
                flat = [int(token) for token in inputs.reshape(-1).tolist()]
                self.target_inputs.append(flat)
                cache[0].offset += len(flat)
                hidden = mx.ones((1, len(flat), 4))
                return self._logits([token + 1 for token in flat]), hidden

            def mtp_prefill(
                self,
                inputs,
                previous_hidden_states,
                cache=None,
                prev_topk_indices=None,
            ):
                self.eval_phase = "mtp"
                flat = [int(token) for token in inputs.reshape(-1).tolist()]
                self.mtp_prefill_inputs.append(flat)
                self.mtp_prefill_prev_topk.append(prev_topk_indices)
                cache.offset += len(flat)
                return mx.ones((1, len(flat), 4)), None

            def mtp_prefill_with_last_logits(
                self,
                inputs,
                previous_hidden_states,
                cache=None,
            ):
                self.eval_phase = "mtp"
                flat = [int(token) for token in inputs.reshape(-1).tolist()]
                self.mtp_last_prefill_inputs.append(flat)
                cache.offset += len(flat)
                hidden = mx.ones((1, 1, 4)) * (flat[-1] + 1)
                return self._logits([flat[-1] + 1]), hidden, mx.array([42])

            def mtp_logits(
                self,
                inputs,
                previous_hidden_states,
                cache=None,
                prev_topk_indices=None,
            ):
                self.eval_phase = "mtp"
                flat = [int(token) for token in inputs.reshape(-1).tolist()]
                self.mtp_logits_inputs.append(flat)
                self.mtp_logits_prev_topk.append(prev_topk_indices)
                cache.offset += len(flat)
                hidden = mx.ones((1, len(flat), 4)) * (flat[-1] + 1)
                return self._logits([flat[-1] + 1]), hidden, mx.array([43])

        model = DummyModel()
        stats = {}
        eval_phases = []
        old_make_cache = generate_module.cache.make_prompt_cache
        old_eval = generate_module.mx.eval
        generate_module.cache.make_prompt_cache = lambda _model: model.target_cache
        generate_module.mx.eval = lambda *args: (
            eval_phases.append(model.eval_phase),
            old_eval(*args),
        )[1]
        try:
            rows = list(
                mtp_speculative_generate_step(
                    mx.array([1, 2, 3]),
                    model,
                    max_tokens=4,
                    num_draft_tokens=2,
                    prefill_step_size=2,
                    mtp_speculative_stats=stats,
                )
            )
        finally:
            generate_module.cache.make_prompt_cache = old_make_cache
            generate_module.mx.eval = old_eval

        self.assertEqual(
            [int(token) for token, _logprobs, _draft in rows],
            [4, 5, 6, 7],
        )
        self.assertEqual(model.mtp_prefill_inputs, [[2, 3], [6]])
        self.assertEqual(model.mtp_last_prefill_inputs, [[4]])
        self.assertEqual(model.mtp_logits_inputs, [[5]])
        self.assertTrue(
            mx.array_equal(model.mtp_logits_prev_topk[0], mx.array([42]))
        )
        self.assertTrue(
            mx.array_equal(model.mtp_prefill_prev_topk[1], mx.array([42]))
        )
        self.assertEqual(model.target_inputs, [[1, 2], [3], [4, 5, 6]])
        self.assertEqual(eval_phases[:4], ["target", "mtp", "target", "mtp"])
        self.assertEqual(stats["mtp_prefill_tokens"], 3)
        self.assertEqual(stats["mtp_prefill_logits_skipped"], 2)
        self.assertTrue(stats["mtp_prefill_shifted"])
        self.assertTrue(stats["mtp_prefill_first_draft_fused"])
        self.assertEqual(stats["drafted_tokens"], 2)
        self.assertEqual(stats["catchup_logits_skipped"], 1)
        self.assertEqual(stats["mtp_iteration_topk_reuses"], 2)

    def test_mtp_speculative_generate_step_verifies_drafts(self):
        class DummyCache:
            def __init__(self):
                self.offset = 0
                self.trimmed = []

            @property
            def state(self):
                return mx.array([self.offset])

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.trimmed.append(n)
                return n

        class DummyModel:
            def __init__(self):
                self.mtp = object()
                self.layers = [object()]
                self.args = type(
                    "Args", (), {"index_share_for_mtp_iteration": True}
                )()
                self.target_cache = [DummyCache()]
                self.mtp_cache = DummyCache()
                self.mtp_topk_reuses = 0
                self.target_next = {
                    3: 5,
                    5: 7,
                    7: 11,
                    99: 101,
                    11: 13,
                    13: 17,
                }
                self.mtp_next = {
                    5: 7,
                    7: 99,
                    11: 13,
                    13: 17,
                }

            def make_mtp_cache(self):
                return self.mtp_cache

            def _logits(self, token, length):
                logits = mx.where(
                    mx.arange(128) == token,
                    mx.array(1.0),
                    mx.array(0.0),
                )
                return mx.broadcast_to(logits.reshape(1, 1, 128), (1, length, 128))

            def forward_with_hidden(self, inputs, cache=None):
                flat = [int(v) for v in inputs.reshape(-1).tolist()]
                outputs = [self.target_next.get(token, 0) for token in flat]
                logits = mx.concatenate(
                    [self._logits(token, 1) for token in outputs],
                    axis=1,
                )
                hidden_values = mx.array(flat, dtype=mx.float32)
                hidden = mx.broadcast_to(
                    hidden_values.reshape(1, len(flat), 1),
                    (1, len(flat), 4),
                )
                return logits, hidden

            def mtp_logits(
                self,
                inputs,
                previous_hidden_states,
                cache=None,
                prev_topk_indices=None,
            ):
                if prev_topk_indices is not None:
                    self.mtp_topk_reuses += 1
                last = int(inputs.reshape(-1)[-1].item())
                token = self.mtp_next.get(last, 0)
                logits = self._logits(token, inputs.shape[1])
                hidden = mx.ones((1, inputs.shape[1], 4)) * token
                return logits, hidden, mx.array([last])

        model = DummyModel()
        stats = {}
        old_make_cache = generate_module.cache.make_prompt_cache
        generate_module.cache.make_prompt_cache = lambda _model: model.target_cache
        try:
            rows = list(
                mtp_speculative_generate_step(
                    mx.array([1, 2, 3]),
                    model,
                    max_tokens=5,
                    num_draft_tokens=3,
                    mtp_speculative_stats=stats,
                )
            )
        finally:
            generate_module.cache.make_prompt_cache = old_make_cache

        self.assertEqual(
            [int(token) for token, _logprobs, _draft in rows],
            [5, 7, 11, 13, 17],
        )
        self.assertEqual(
            [draft for _token, _logprobs, draft in rows],
            [False, True, False, True, False],
        )
        self.assertIn(2, model.target_cache[0].trimmed)
        self.assertIn(1, model.mtp_cache.trimmed)
        self.assertEqual(stats["rounds"], 2)
        self.assertEqual(stats["drafted_tokens"], 4)
        self.assertEqual(stats["draft_logsumexp_skipped"], 4)
        self.assertEqual(stats["accepted_tokens"], 2)
        self.assertEqual(stats["target_forwards"], 2)
        self.assertEqual(stats["target_input_tokens"], 6)
        self.assertEqual(stats["target_greedy_verify_batches"], 2)
        self.assertEqual(stats["target_greedy_verify_tokens"], 6)
        self.assertEqual(stats["mtp_iteration_topk_reuses"], 3)
        self.assertEqual(model.mtp_topk_reuses, 3)
        self.assertEqual(stats["target_logsumexp_skipped"], 0)
        self.assertTrue(stats["return_logprobs"])
        self.assertEqual(stats["target_tokens"], 3)
        self.assertEqual(stats["emitted_tokens"], 5)
        self.assertEqual(stats["catchup_forwards"], 1)
        self.assertAlmostEqual(stats["acceptance_rate"], 0.5)
        self.assertAlmostEqual(stats["mean_accepted"], 1.0)
        self.assertAlmostEqual(stats["emitted_per_target_forward"], 2.5)

    def test_mtp_speculative_skips_unrequested_greedy_target_logprobs(self):
        class DummyCache:
            def __init__(self):
                self.offset = 0

            @property
            def state(self):
                return mx.array([self.offset])

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.offset = max(0, self.offset - n)
                return n

        class DummyModel:
            def __init__(self):
                self.mtp = object()
                self.layers = [object()]
                self.target_cache = [DummyCache()]
                self.mtp_cache = DummyCache()

            def make_mtp_cache(self):
                return self.mtp_cache

            def _logits(self, token, length):
                logits = mx.where(
                    mx.arange(32) == token,
                    mx.array(1.0),
                    mx.array(0.0),
                )
                return mx.broadcast_to(logits.reshape(1, 1, 32), (1, length, 32))

            def forward_with_hidden(self, inputs, cache=None):
                length = inputs.shape[1]
                logits = self._logits(5, length)
                hidden = mx.ones((1, length, 4))
                return logits, hidden

            def mtp_logits(self, inputs, previous_hidden_states, cache=None):
                length = inputs.shape[1]
                return self._logits(5, length), mx.ones((1, length, 4)), None

        model = DummyModel()
        stats = {}
        old_make_cache = generate_module.cache.make_prompt_cache
        generate_module.cache.make_prompt_cache = lambda _model: model.target_cache
        try:
            rows = list(
                mtp_speculative_generate_step(
                    mx.array([1, 2, 3]),
                    model,
                    max_tokens=4,
                    num_draft_tokens=2,
                    mtp_speculative_stats=stats,
                    mtp_return_logprobs=False,
                )
            )
        finally:
            generate_module.cache.make_prompt_cache = old_make_cache

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(logprobs is None for _token, logprobs, _draft in rows))
        self.assertFalse(stats["return_logprobs"])
        self.assertEqual(stats["target_logsumexp_skipped"], 4)

    def test_mtp_speculative_falls_back_after_low_acceptance(self):
        class DummyCache:
            def __init__(self):
                self.offset = 0

            @property
            def state(self):
                return mx.array([self.offset])

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.offset = max(0, self.offset - n)
                return n

        class DummyModel:
            def __init__(self):
                self.mtp = object()
                self.layers = [object()]
                self.target_cache = [DummyCache()]
                self.mtp_cache = DummyCache()
                self.mtp_calls = 0
                self.mtp_prefill_calls = 0

            def make_mtp_cache(self):
                return self.mtp_cache

            def _logits(self, tokens):
                rows = []
                for token in tokens:
                    logits = mx.where(
                        mx.arange(32) == token,
                        mx.array(1.0),
                        mx.array(0.0),
                    )
                    rows.append(logits.reshape(1, 1, 32))
                return mx.concatenate(rows, axis=1)

            def forward_with_hidden(self, inputs, cache=None):
                flat = [int(token) for token in inputs.reshape(-1).tolist()]
                for target_cache in cache or []:
                    target_cache.offset += len(flat)
                outputs = [min(token + 1, 31) for token in flat]
                hidden = mx.ones((1, len(flat), 4))
                return self._logits(outputs), hidden

            def mtp_logits(self, inputs, previous_hidden_states, cache=None):
                self.mtp_calls += 1
                length = inputs.shape[1]
                cache.offset += length
                hidden = mx.ones((1, length, 4))
                return self._logits([31] * length), hidden, None

            def mtp_prefill(self, inputs, previous_hidden_states, cache=None):
                self.mtp_prefill_calls += 1
                cache.offset += inputs.shape[1]
                return mx.ones((1, inputs.shape[1], 4)), None

        model = DummyModel()
        stats = {}
        old_make_cache = generate_module.cache.make_prompt_cache
        generate_module.cache.make_prompt_cache = lambda _model: model.target_cache
        try:
            rows = list(
                mtp_speculative_generate_step(
                    mx.array([1]),
                    model,
                    max_tokens=5,
                    num_draft_tokens=2,
                    mtp_speculative_stats=stats,
                    mtp_adaptive_fallback_min_drafted_tokens=2,
                    mtp_adaptive_fallback_min_acceptance_rate=0.5,
                )
            )
        finally:
            generate_module.cache.make_prompt_cache = old_make_cache

        self.assertEqual(
            [int(token) for token, _logprobs, _draft in rows],
            [2, 3, 4, 5, 6],
        )
        self.assertEqual(model.mtp_calls, 2)
        self.assertEqual(model.mtp_prefill_calls, 4)
        self.assertTrue(stats["adaptive_fallback"])
        self.assertEqual(stats["adaptive_fallback_at_emitted_tokens"], 2)
        self.assertEqual(stats["adaptive_fallback_acceptance_rate"], 0.0)
        self.assertEqual(stats["adaptive_fallback_target_forwards"], 3)
        self.assertEqual(stats["adaptive_fallback_mtp_cache_forwards"], 3)
        self.assertEqual(stats["adaptive_fallback_mtp_logits_skipped"], 3)
        self.assertEqual(stats["rounds"], 1)
        self.assertEqual(stats["drafted_tokens"], 2)
        self.assertEqual(stats["accepted_tokens"], 0)
        self.assertEqual(model.target_cache[0].offset, 5)
        self.assertEqual(model.mtp_cache.offset, 5)

    def test_mtp_speculative_generate_step_uses_combined_prompt_cache(self):
        class DummyCache:
            def __init__(self, offset=0):
                self.offset = offset

            @property
            def state(self):
                return mx.array([self.offset])

            def size(self):
                return self.offset

            def is_trimmable(self):
                return True

            def trim(self, n):
                n = min(n, self.offset)
                self.offset -= n
                return n

        class DummyModel:
            def __init__(self):
                self.mtp = object()
                self.layers = [object()]
                self.target_cache = [DummyCache(offset=2)]
                self.mtp_cache = DummyCache(offset=2)
                self.target_next = {
                    3: 5,
                    5: 7,
                    7: 11,
                }
                self.mtp_next = {
                    5: 7,
                    7: 11,
                }

            def make_cache(self):
                return [DummyCache()]

            def make_mtp_cache(self):
                return DummyCache()

            def _logits(self, token, length):
                logits = mx.where(
                    mx.arange(128) == token,
                    mx.array(1.0),
                    mx.array(0.0),
                )
                return mx.broadcast_to(logits.reshape(1, 1, 128), (1, length, 128))

            def forward_with_hidden(self, inputs, cache=None):
                flat = [int(v) for v in inputs.reshape(-1).tolist()]
                if cache is not None:
                    for c in cache:
                        c.offset += len(flat)
                outputs = [self.target_next.get(token, 0) for token in flat]
                logits = mx.concatenate(
                    [self._logits(token, 1) for token in outputs],
                    axis=1,
                )
                hidden_values = mx.array(flat, dtype=mx.float32)
                hidden = mx.broadcast_to(
                    hidden_values.reshape(1, len(flat), 1),
                    (1, len(flat), 4),
                )
                return logits, hidden

            def mtp_logits(self, inputs, previous_hidden_states, cache=None):
                if cache is not None:
                    cache.offset += inputs.shape[1]
                last = int(inputs.reshape(-1)[-1].item())
                token = self.mtp_next.get(last, 0)
                logits = self._logits(token, inputs.shape[1])
                hidden = mx.ones((1, inputs.shape[1], 4)) * token
                return logits, hidden, None

        model = DummyModel()
        stats = {}
        progress = []
        rows = list(
            mtp_speculative_generate_step(
                mx.array([3]),
                model,
                prompt_cache=[model.target_cache[0], model.mtp_cache],
                prompt_history=mx.array([1, 2, 3]),
                max_tokens=3,
                num_draft_tokens=2,
                mtp_speculative_stats=stats,
                prompt_progress_callback=lambda processed, total: progress.append(
                    (processed, total)
                ),
            )
        )

        self.assertEqual(
            [int(token) for token, _logprobs, _draft in rows],
            [5, 7, 11],
        )
        self.assertEqual(
            [draft for _token, _logprobs, draft in rows],
            [False, True, False],
        )
        self.assertEqual(progress[0], (2, 3))
        self.assertEqual(progress[-1], (3, 3))
        self.assertEqual(stats["cached_prompt_tokens"], 2)
        self.assertEqual(stats["fresh_prompt_tokens"], 1)
        self.assertEqual(model.target_cache[0].offset, 5)
        self.assertEqual(model.mtp_cache.offset, 5)

    def test_mtp_speculative_generate_step_trims_exact_prompt_cache(self):
        class DummyCache:
            def __init__(self, offset=0):
                self.offset = offset

            @property
            def state(self):
                return mx.array([self.offset])

            def size(self):
                return self.offset

            def is_trimmable(self):
                return True

            def trim(self, n):
                n = min(n, self.offset)
                self.offset -= n
                return n

        class DummyModel:
            def __init__(self):
                self.mtp = object()
                self.layers = [object()]
                self.target_cache = [DummyCache(offset=3)]
                self.mtp_cache = DummyCache(offset=3)
                self.mtp_last_prefill_inputs = []

            def make_cache(self):
                return [DummyCache()]

            def make_mtp_cache(self):
                return DummyCache()

            def _logits(self, token, length):
                logits = mx.where(
                    mx.arange(128) == token,
                    mx.array(1.0),
                    mx.array(0.0),
                )
                return mx.broadcast_to(logits.reshape(1, 1, 128), (1, length, 128))

            def forward_with_hidden(self, inputs, cache=None):
                flat = [int(v) for v in inputs.reshape(-1).tolist()]
                if cache is not None:
                    for c in cache:
                        c.offset += len(flat)
                logits = mx.concatenate(
                    [self._logits(5 if token == 3 else 0, 1) for token in flat],
                    axis=1,
                )
                hidden_values = mx.array(flat, dtype=mx.float32)
                hidden = mx.broadcast_to(
                    hidden_values.reshape(1, len(flat), 1),
                    (1, len(flat), 4),
                )
                return logits, hidden

            def mtp_logits(self, inputs, previous_hidden_states, cache=None):
                if cache is not None:
                    cache.offset += inputs.shape[1]
                logits = self._logits(0, inputs.shape[1])
                hidden = mx.ones((1, inputs.shape[1], 4))
                return logits, hidden, None

            def mtp_prefill_with_last_logits(
                self,
                inputs,
                previous_hidden_states,
                cache=None,
            ):
                flat = [int(v) for v in inputs.reshape(-1).tolist()]
                self.mtp_last_prefill_inputs.append(flat)
                cache.offset += len(flat)
                logits = self._logits(0, 1)
                hidden = mx.ones((1, 1, 4))
                return logits, hidden, None

        model = DummyModel()
        stats = {}
        progress = []
        rows = list(
            mtp_speculative_generate_step(
                mx.array([], dtype=mx.uint32),
                model,
                prompt_cache=[model.target_cache[0], model.mtp_cache],
                prompt_history=mx.array([1, 2, 3]),
                max_tokens=1,
                mtp_speculative_stats=stats,
                prompt_progress_callback=lambda processed, total: progress.append(
                    (processed, total)
                ),
            )
        )

        self.assertEqual([int(token) for token, _logprobs, _draft in rows], [5])
        self.assertEqual(progress[0], (2, 3))
        self.assertEqual(progress[-1], (3, 3))
        self.assertEqual(stats["cached_prompt_tokens"], 2)
        self.assertEqual(stats["fresh_prompt_tokens"], 1)
        self.assertEqual(model.target_cache[0].offset, 3)
        self.assertEqual(model.mtp_cache.offset, 3)
        self.assertEqual(model.mtp_last_prefill_inputs, [[5]])
        self.assertTrue(stats["mtp_prefill_shifted"])
        self.assertTrue(stats["mtp_prefill_first_draft_fused"])

    def test_mtp_speculative_generate_step_syncs_quantized_combined_cache(self):
        class DummyCache:
            def __init__(self, offset=0, quantized=False):
                self.offset = offset
                self.quantized = quantized

            @property
            def state(self):
                return mx.array([self.offset])

            def size(self):
                return self.offset

            def is_trimmable(self):
                return True

            def trim(self, n):
                n = min(n, self.offset)
                self.offset -= n
                return n

            def to_quantized(self, group_size=64, bits=8):
                return DummyCache(offset=self.offset, quantized=True)

        class DummyModel:
            def __init__(self):
                self.mtp = object()
                self.layers = [object()]
                self.target_prefill_calls = 0
                self.mtp_prefill_calls = 0
                self.mtp_logits_calls = 0

            def make_cache(self):
                return [DummyCache()]

            def make_mtp_cache(self):
                return DummyCache()

            def _logits(self, token, length):
                logits = mx.where(
                    mx.arange(16) == token,
                    mx.array(1.0),
                    mx.array(0.0),
                )
                return mx.broadcast_to(logits.reshape(1, 1, 16), (1, length, 16))

            def forward_with_hidden(self, inputs, cache=None):
                for c in cache:
                    c.offset += inputs.shape[1]
                hidden = mx.ones((1, inputs.shape[1], 4))
                return self._logits(5, inputs.shape[1]), hidden

            def prefill_with_hidden(self, inputs, cache=None):
                self.target_prefill_calls += 1
                logits, hidden = self.forward_with_hidden(inputs, cache=cache)
                return logits[:, -1:, :], hidden

            def mtp_logits(self, inputs, previous_hidden_states, cache=None):
                self.mtp_logits_calls += 1
                cache.offset += inputs.shape[1]
                hidden = mx.ones((1, inputs.shape[1], 4))
                return self._logits(6, inputs.shape[1]), hidden, None

            def mtp_prefill(self, inputs, previous_hidden_states, cache=None):
                self.mtp_prefill_calls += 1
                cache.offset += inputs.shape[1]
                hidden = mx.ones((1, inputs.shape[1], 4))
                return hidden, None

        model = DummyModel()
        combined_cache = [DummyCache(), DummyCache()]
        stats = {}
        rows = list(
            mtp_speculative_generate_step(
                mx.array([1, 2, 3]),
                model,
                prompt_cache=combined_cache,
                max_tokens=1,
                prefill_step_size=2,
                kv_bits=8,
                quantized_kv_start=0,
                mtp_speculative_stats=stats,
            )
        )

        self.assertEqual([token for token, _logprobs, _draft in rows], [5])
        self.assertTrue(combined_cache[0].quantized)
        self.assertTrue(combined_cache[1].quantized)
        self.assertEqual(combined_cache[0].offset, 3)
        self.assertEqual(combined_cache[1].offset, 3)
        self.assertEqual(model.target_prefill_calls, 2)
        self.assertEqual(model.mtp_prefill_calls, 2)
        self.assertEqual(model.mtp_logits_calls, 0)
        self.assertEqual(stats["target_prefill_tokens"], 3)
        self.assertEqual(stats["target_prefill_logits_skipped"], 1)
        self.assertEqual(stats["mtp_prefill_tokens"], 3)
        self.assertEqual(stats["mtp_prefill_logits_skipped"], 3)

    def test_effective_prefill_step_size_caps_long_context(self):
        step = _effective_prefill_step_size(
            requested_step_size=1024,
            remaining_tokens=2000,
            processed_tokens=200_000,
            prefill_max_qk_tokens=67_108_864,
        )

        self.assertLess(step, 1024)
        self.assertLessEqual(step * (200_000 + step), 67_108_864)

    def test_effective_prefill_step_size_can_be_disabled(self):
        step = _effective_prefill_step_size(
            requested_step_size=1024,
            remaining_tokens=2000,
            processed_tokens=200_000,
            prefill_max_qk_tokens=0,
        )

        self.assertEqual(step, 1024)

    def test_glm_dsa_adaptive_prefill_step_is_opt_in(self):
        model = type("Model", (), {"config": {"model_type": "glm_moe_dsa"}})()

        step = _glm_dsa_adaptive_prefill_step_size(
            model,
            requested_step_size=1024,
            processed_tokens=0,
            remaining_tokens=20_000,
            adaptive_step_size=8192,
        )

        self.assertEqual(step, 8192)

    def test_glm_dsa_adaptive_prefill_step_respects_qk_cap(self):
        model = type("Model", (), {"config": {"model_type": "glm_moe_dsa"}})()
        adaptive_step = _glm_dsa_adaptive_prefill_step_size(
            model,
            requested_step_size=1024,
            processed_tokens=200_000,
            remaining_tokens=20_000,
            adaptive_step_size=8192,
        )

        step = _effective_prefill_step_size(
            requested_step_size=1024,
            remaining_tokens=20_000,
            processed_tokens=200_000,
            prefill_max_qk_tokens=67_108_864,
            adaptive_step_size=adaptive_step,
        )

        self.assertLess(step, 8192)
        self.assertLessEqual(step * (200_000 + step), 67_108_864)

    def test_glm_dsa_adaptive_prefill_step_skips_non_glm(self):
        model = type("Model", (), {"config": {"model_type": "llama"}})()

        step = _glm_dsa_adaptive_prefill_step_size(
            model,
            requested_step_size=1024,
            processed_tokens=0,
            remaining_tokens=20_000,
            adaptive_step_size=8192,
        )

        self.assertIsNone(step)

    def test_glm_dsa_prefill_chunk_fields_report_deltas(self):
        before = {
            "fast_prefill_hits": 0,
            "native_sparse_prefill_hits": 10,
            "native_indexer_hits": 4,
            "fallback_reasons": {},
            "native_sparse_prefill_fallback_reasons": {"below": 1},
            "native_indexer_fallback_reasons": {},
            "stages": {
                "native_sparse_attention": {"seconds": 1.0, "count": 2},
                "native_indexer_scores": {"seconds": 0.25, "count": 1},
            },
        }
        after = {
            "fast_prefill_hits": 0,
            "native_sparse_prefill_hits": 88,
            "native_indexer_hits": 25,
            "fallback_reasons": {},
            "native_sparse_prefill_fallback_reasons": {
                "below": 1,
                "runtime_error": 2,
            },
            "native_indexer_fallback_reasons": {},
            "stages": {
                "native_sparse_attention": {"seconds": 1.5, "count": 3},
                "native_indexer_scores": {"seconds": 0.375, "count": 2},
            },
        }

        fields = _glm_dsa_prefill_profile_chunk_fields(before, after)

        self.assertEqual(fields["glm_dsa_native_sparse_prefill_hits"], 78)
        self.assertEqual(fields["glm_dsa_native_indexer_hits"], 21)
        self.assertEqual(
            fields["glm_dsa_native_sparse_prefill_fallback_reasons"],
            {"runtime_error": 2},
        )
        self.assertEqual(fields["glm_dsa_native_sparse_attention_seconds"], 0.5)
        self.assertEqual(fields["glm_dsa_native_sparse_attention_count"], 1)
        self.assertEqual(fields["glm_dsa_native_indexer_scores_seconds"], 0.125)
        self.assertIn(
            'glm_dsa_native_sparse_prefill_fallback_reasons={"runtime_error":2}',
            _format_prefill_chunk_fields(fields),
        )

    def test_glm_dsa_decode_profile_fields_report_deltas(self):
        before = {
            "stages": {
                "q_projection": {"seconds": 1.0, "count": 10},
                "mlp": {"seconds": 2.0, "count": 10},
                "total_decode_model": {"seconds": 3.0, "count": 5},
            },
        }
        after = {
            "stages": {
                "q_projection": {"seconds": 1.25, "count": 12},
                "mlp": {"seconds": 2.75, "count": 12},
                "total_decode_model": {"seconds": 3.5, "count": 6},
            },
        }

        fields = _glm_dsa_decode_profile_fields(before, after)

        self.assertEqual(fields["glm_dsa_decode_q_projection_seconds"], 0.25)
        self.assertEqual(fields["glm_dsa_decode_q_projection_count"], 2)
        self.assertEqual(fields["glm_dsa_decode_mlp_seconds"], 0.75)
        self.assertEqual(fields["glm_dsa_decode_mlp_count"], 2)
        self.assertEqual(fields["glm_dsa_decode_total_decode_model_seconds"], 0.5)
        self.assertEqual(fields["glm_dsa_decode_total_decode_model_count"], 1)
        self.assertIn(
            "glm_dsa_decode_mlp_seconds=0.750000",
            _format_prefill_chunk_fields(fields),
        )


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def test_generate(self):
        # Simple test that generation runs
        text = generate(
            self.model, self.tokenizer, "hello", max_tokens=5, verbose=False
        )

    def test_generate_with_logit_bias(self):
        logit_bias = {0: 2000.0, 1: -20.0}
        text = generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            logits_processors=make_logits_processors(logit_bias),
            verbose=False,
        )
        self.assertEqual(text, "!!!!!")

    def test_stream_generate_max_tokens(self):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a story about Einstein"}],
            tokenize=True,
            add_generation_prompt=True,
        )

        tokens = []
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
        ):
            tokens.append(response.token)
        self.assertEqual(len(tokens), 4)

    def test_generate_with_processor(self):
        init_toks = self.tokenizer.encode("hello")

        all_toks = None

        def logits_processor(toks, logits):
            nonlocal all_toks
            all_toks = toks
            return logits

        generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            verbose=False,
            logits_processors=[logits_processor],
        )
        self.assertEqual(len(all_toks), len(init_toks) + 5)

    def test_stream_generate_speculative(self):
        # Use same model as draft model, this is not a speed test
        draft_model = self.model

        results: List[GenerationResponse] = []
        drafted: List[bool] = []

        # make a determinate sampler
        sampler = make_sampler(temp=0.0)
        messages = [{"role": "user", "content": "hello"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
        ):
            drafted.append(generation_result.from_draft)
            results.append(generation_result)

        self.assertEqual(len(results), 5)
        # since num_draft_tokens is 2 and draft model is the same, the
        # first 2 generations should be drafts, the third should come
        # from the target model, and last two should be drafts
        self.assertEqual(drafted, [True, True, False, True, True])

    def test_stream_generate_input_embeddings(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)

    def test_stream_generate_input_embeddings_prefill(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        # setup prompt progress callback to track batched prefill
        num_prompt_processing_callbacks = 0

        def progress_callback(processed: int, total: int) -> None:
            nonlocal num_prompt_processing_callbacks
            num_prompt_processing_callbacks += 1

        # generate
        prefill_step_size = 5
        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
            prefill_step_size=prefill_step_size,
            prompt_progress_callback=progress_callback,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)
        num_embeddings = prompt_embeddings.shape[0]
        self.assertTrue(
            num_embeddings / prefill_step_size < num_prompt_processing_callbacks
        )

    def test_batch_matches_single(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model, stop_tokens=self.tokenizer.eos_token_ids, max_tokens=1
        )
        uids = gen.insert(prompts)
        batch_responses = {r.uid: r for r in gen.next_generated()}

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_many_batches(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=1,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        uids = gen.insert(prompts)
        batch_responses = {}
        not_in = True
        iters = 0
        while responses := gen.next_generated():
            for r in responses:
                not_in &= r.uid not in batch_responses
                batch_responses[r.uid] = r
            iters += 1
        # only one token per prompt means only one response per prompt
        self.assertTrue(not_in)

        # completion batch size is too small for a single iteration
        self.assertTrue(iters > 1)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_batch_unique_max_toks(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        num_toks = [2, 3, 4, 5]
        uids = gen.insert(prompts, max_tokens=num_toks)
        batch_responses = {uid: [] for uid in uids}
        while responses := gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.token)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            tokens = []
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=num_toks[e],
            ):
                tokens.append(response.token)

            batch_tokens = batch_responses[uids[e]]
            self.assertEqual(tokens, batch_tokens)

    def test_batch_sliding_window(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        self.model.make_cache = lambda: [
            RotatingKVCache(max_size=4) for _ in self.model.layers
        ]
        batch_gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=10,
            prefill_batch_size=1,
            prefill_step_size=8,
            completion_batch_size=2,
        )
        uids = batch_gen.insert(prompts)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, response in enumerate(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    prompts[e],
                    max_tokens=10,
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                logprobs = response.logprobs
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

        del self.model.make_cache

    def test_batch_generate_with_logits_processors(self):
        """Test that batch_generate with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        logprobs = response.logprobs
        self.assertEqual(logprobs[0].item(), 0.0)
        self.assertEqual(logprobs.argmin().item(), 1)

        del batch_gen

        logit_bias = {0: 2000.0}
        processors = make_logits_processors(logit_bias)
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )

        (uid0,) = batch_gen.insert([prompt])

        logit_bias = {1: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid1,) = batch_gen.insert([prompt], logits_processors=[processors])

        logit_bias = {2: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid2,) = batch_gen.insert([prompt], logits_processors=[processors])

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].logprobs[0].item(), 0.0)
        self.assertEqual(responses[uid1].logprobs[1].item(), 0.0)
        self.assertEqual(responses[uid2].logprobs[2].item(), 0.0)

    def test_batch_generate_processor_tokens_match_prompt_on_first_step(self):
        prompt = self.tokenizer.encode("hello")
        seen = []

        def processor(tokens, logits):
            seen.append(tokens)
            return logits

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=[processor],
        )
        batch_gen.insert([prompt])
        batch_gen.next_generated()

        self.assertTrue(hasattr(seen[0], "shape"))
        self.assertEqual(seen[0].tolist(), prompt)

    def test_batch_generate_function_with_logits_processors(self):
        """Test that batch_generate function with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        prompts = [self.tokenizer.encode("hello")]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=1,
            logits_processors=processors,
        )
        self.assertEqual(len(response.texts), 1)
        generated_token = self.tokenizer.encode(response.texts[0])[0]
        self.assertEqual(generated_token, 0)

    def test_batch_generate_with_samplers(self):
        """Test that batch_generate with logits_processors produces correct results."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next_generated()[0]
        self.assertEqual(response.token, 1)

        del batch_gen

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )

        (uid0,) = batch_gen.insert([prompt])
        uid1, uid2 = batch_gen.insert(
            [prompt, prompt],
            samplers=[lambda _: mx.array([2]), lambda _: mx.array([3])],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].token, 1)
        self.assertEqual(responses[uid1].token, 2)
        self.assertEqual(responses[uid2].token, 3)

    def test_batch_generate_with_state_machines(self):
        """Test that batch_generate with per-sequence state_machines stops on different tokens."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=10,
        )
        prompt = self.tokenizer.encode("hello")

        sm_0 = SequenceStateMachine({"normal": [([0], None)]}, initial="normal")
        sm_1 = SequenceStateMachine({"normal": [([1], None)]}, initial="normal")
        sm_2 = SequenceStateMachine({"normal": [([2], None)]}, initial="normal")

        processor_0 = make_logits_processors({0: 2000.0})
        processor_1 = make_logits_processors({1: 2000.0})
        processor_2 = make_logits_processors({2: 2000.0})

        uid0, uid1, uid2 = batch_gen.insert(
            [prompt, prompt, prompt],
            logits_processors=[processor_0, processor_1, processor_2],
            state_machines=[sm_0, sm_1, sm_2],
        )

        responses = batch_gen.next_generated()
        responses = {response.uid: response for response in responses}

        self.assertEqual(responses[uid0].token, 0)
        self.assertEqual(responses[uid1].token, 1)
        self.assertEqual(responses[uid2].token, 2)
        self.assertEqual(responses[uid0].finish_reason, "stop")
        self.assertEqual(responses[uid1].finish_reason, "stop")
        self.assertEqual(responses[uid2].finish_reason, "stop")
        self.assertEqual(responses[uid0].match_sequence, (0,))
        self.assertEqual(responses[uid1].match_sequence, (1,))
        self.assertEqual(responses[uid2].match_sequence, (2,))

    def test_batch_continued_generation(self):
        for rotating in [False, True]:
            if rotating:
                self.model.make_cache = lambda: [
                    RotatingKVCache(max_size=4) for _ in self.model.layers
                ]

            # Make the prompts
            prompts_a = [
                "Write a story about Einstein",
                "Hi",
                "What time is it?",
                "How tall is Mt Everest?",
            ]
            prompts_a = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_a
            ]
            prompts_b = [
                "Another one",
                "sup?",
                "And how about the date?",
                "Mt Olympus?",
            ]
            prompts_b = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_b
            ]

            # Generate once
            batch_gen = BatchGenerator(
                self.model,
                stop_tokens=self.tokenizer.eos_token_ids,
                max_tokens=10,
                prefill_batch_size=4,
                prefill_step_size=8,
                completion_batch_size=2,
            )
            uids = batch_gen.insert(prompts_a)
            caches = {uid: None for uid in uids}
            while responses := batch_gen.next_generated():
                for r in responses:
                    if r.finish_reason is not None:
                        caches[r.uid] = r.prompt_cache
            caches = [caches[uid] for uid in uids]

            # Generate the 2nd time
            uids = batch_gen.insert(prompts_b, caches=caches)
            batch_responses = {uid: [] for uid in uids}
            while responses := batch_gen.next_generated():
                for r in responses:
                    batch_responses[r.uid].append(r.logprobs)

            for e, uid in enumerate(uids):
                for i, response in enumerate(
                    stream_generate(
                        self.model,
                        self.tokenizer,
                        prompts_b[e],
                        max_tokens=10,
                        prompt_cache=caches[e],
                    )
                ):
                    batch_logprobs = batch_responses[uid][i]
                    logprobs = response.logprobs
                    self.assertTrue(
                        mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                    )

            if rotating:
                del self.model.make_cache

    def _continued_generation_test_helper(self, model):
        def rand_prompt(n):
            return [random.randint(0, 1000) for _ in range(n)]

        # Make the prompts
        prompts_a = [
            rand_prompt(5),
            rand_prompt(3),
            rand_prompt(8),
            rand_prompt(1),
        ]
        prompts_b = [
            rand_prompt(2),
            rand_prompt(7),
            rand_prompt(4),
            rand_prompt(6),
        ]

        # Generate once
        batch_gen = BatchGenerator(
            model,
            stop_tokens={},
            max_tokens=10,
            prefill_batch_size=4,
            prefill_step_size=32,
            completion_batch_size=2,
        )

        uids = batch_gen.insert(prompts_a)
        caches = {uid: None for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    caches[r.uid] = r.prompt_cache

        caches = [caches[uid] for uid in uids]

        # Generate the 2nd time
        uids = batch_gen.insert(prompts_b, caches=caches)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next_generated():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, (_, logprobs) in enumerate(
                generate_step(
                    mx.array(prompts_b[e]),
                    model,
                    max_tokens=10,
                    prompt_cache=caches[e],
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

    def test_batch_continued_generation_ssm(self):
        from mlx_lm.models import mamba2

        random.seed(0)
        mx.random.seed(4)

        # Make a small SSM model
        args = mamba2.ModelArgs(
            model_type="mamba2",
            num_heads=8,
            head_dim=16,
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=128,
            state_size=32,
            num_hidden_layers=4,
            layer_norm_epsilon=1e-4,
            conv_kernel=3,
            n_groups=4,
            use_bias=False,
            use_conv_bias=False,
            tie_word_embeddings=True,
            time_step_limit=(0.01, 10),
            time_step_rank="auto",
        )
        model = mamba2.Model(args)
        self._continued_generation_test_helper(model)

    def test_batch_continued_generation_gated_delta(self):
        from mlx_lm.models import qwen3_next

        random.seed(0)
        mx.random.seed(4)
        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=128,
            mlp_only_layers=[0],
            moe_intermediate_size=128,
            rms_norm_eps=1e-5,
            head_dim=64,
            rope_theta=1000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=1000,
        )
        model = qwen3_next.Model(args)
        self._continued_generation_test_helper(model)

    def test_extend_cache_with_empty(self):
        from mlx_lm.generate import _extend_cache
        from mlx_lm.models.cache import make_prompt_cache

        cache_a = make_prompt_cache(self.model)

        prompt = mx.array([[1, 2, 3]])
        self.model(prompt, cache=cache_a)
        mx.eval([c.state for c in cache_a])

        result = _extend_cache(cache_a, [])
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

        result = _extend_cache([], cache_a)
        self.assertEqual(len(result), len(cache_a))
        for c in result:
            self.assertGreater(c.offset, 0)

    def test_remove_prompt_batch_updates_currently_processing(self):
        prompt_a = self.tokenizer.encode("Write a long story about a cat")
        prompt_b = self.tokenizer.encode("Write a long story about a dog")

        gen = BatchGenerator(
            self.model,
            max_tokens=5,
            prefill_batch_size=2,
            prefill_step_size=4,
            completion_batch_size=4,
        )
        uid_a, uid_b = gen.insert([prompt_a, prompt_b])

        gen.next()

        found = gen._find_uids([uid_a, uid_b])
        for uid in [uid_a, uid_b]:
            self.assertIn(uid, found)
            self.assertEqual(found[uid][0], 1)

        gen.remove([uid_a])

        self.assertEqual(len(gen._currently_processing), len(gen._prompt_batch))

        found = gen._find_uids([uid_b])
        self.assertIn(uid_b, found)

        while responses := gen.next_generated():
            if all(r.finish_reason is not None for r in responses):
                break

    def test_batch_max_kv_size_creates_rotating_cache(self):
        max_kv_size = 256
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=max_kv_size,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, RotatingKVCache)
                    self.assertEqual(cache.max_size, max_kv_size)

    def test_batch_max_kv_size_limits_cache_growth(self):
        max_kv_size = 5
        gen = BatchGenerator(
            self.model,
            max_tokens=10,
            max_kv_size=max_kv_size,
            prefill_batch_size=1,
            prefill_step_size=128,
            completion_batch_size=1,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertLessEqual(cache.keys.shape[2], max_kv_size)

    def test_batch_max_kv_size_none_creates_regular_cache(self):
        gen = BatchGenerator(
            self.model,
            max_tokens=1,
            max_kv_size=None,
        )

        prompt = self.tokenizer.encode("Write a long story about a cat")
        gen.insert([prompt])

        for r in gen.next_generated():
            if r.finish_reason is not None:
                for cache in r.prompt_cache:
                    self.assertIsInstance(cache, KVCache)

    def test_batch_generate_return_logprobs(self):
        """Test that batch_generate returns per-token logprobs when requested."""
        prompts = [
            self.tokenizer.encode("hello"),
            self.tokenizer.encode("write a poem"),
        ]
        max_tokens = 5
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=max_tokens,
            return_logprobs=True,
            return_token_ids=True,
        )

        # Check that logprobs and token_ids are returned
        self.assertIsNotNone(response.logprobs)
        self.assertIsNotNone(response.token_ids)
        self.assertEqual(len(response.logprobs), len(prompts))
        self.assertEqual(len(response.token_ids), len(prompts))

        for i in range(len(prompts)):
            # token_ids and logprobs should have same length
            self.assertEqual(len(response.token_ids[i]), len(response.logprobs[i]))
            # logprobs should be non-positive (log-probabilities)
            for lp in response.logprobs[i]:
                self.assertLessEqual(lp, 0.0)

    def test_batch_generate_no_logprobs_by_default(self):
        """Test that batch_generate does not return logprobs by default."""
        prompts = [self.tokenizer.encode("hello")]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=3,
        )
        self.assertIsNone(response.logprobs)
        self.assertIsNone(response.token_ids)


if __name__ == "__main__":
    unittest.main()
