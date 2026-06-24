# Copyright © 2025 Apple Inc.

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, GlmMlaKVCache, KVCache, QuantizedGlmMlaKVCache
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from .deepseek_v32 import Model as DSV32Model


GLM_DSA_FAST_PREFILL_ENV = "MLX_LM_GLM_DSA_FAST_PREFILL"
GLM_DSA_FAST_PREFILL_DEBUG_ENV = "MLX_LM_GLM_DSA_FAST_PREFILL_DEBUG"
GLM_DSA_FAST_PREFILL_QUERY_CHUNK_ENV = "MLX_LM_GLM_DSA_FAST_PREFILL_QUERY_CHUNK"
GLM_DSA_PREFILL_PROFILE_ENV = "MLX_LM_GLM_DSA_PREFILL_PROFILE"

_PROFILE_STAGES = (
    "q_projection",
    "kv_cache_update",
    "dsa_indexer_topk",
    "latent_kv_dequantization",
    "latent_kv_projection",
    "sparse_gather",
    "attention",
    "total_prefill",
)
_DEFAULT_FAST_PREFILL_QUERY_CHUNK = 16
_FAST_PREFILL_LARGE_TOPK_WARNING = 1024
_LOGGER = logging.getLogger(__name__)
_GLM_DSA_PREFILL_PROFILE = None
_WARNED_FAST_PREFILL_LARGE_TOPK = False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _fast_prefill_enabled() -> bool:
    return _env_flag(GLM_DSA_FAST_PREFILL_ENV, False)


def _prefill_profile_enabled() -> bool:
    return _env_flag(GLM_DSA_PREFILL_PROFILE_ENV, False)


def _fast_prefill_debug_enabled() -> bool:
    return _env_flag(GLM_DSA_FAST_PREFILL_DEBUG_ENV, False)


def _fast_prefill_query_chunk_size(sequence_length: int) -> int:
    raw_value = os.environ.get(GLM_DSA_FAST_PREFILL_QUERY_CHUNK_ENV)
    if raw_value is not None:
        try:
            return max(1, min(sequence_length, int(raw_value)))
        except ValueError:
            pass
    return min(sequence_length, _DEFAULT_FAST_PREFILL_QUERY_CHUNK)


def _new_profile():
    return {
        "stages": {
            stage: {"seconds": 0.0, "count": 0} for stage in _PROFILE_STAGES
        },
        "fast_prefill_hits": 0,
        "fallback_reasons": Counter(),
    }


def reset_glm_dsa_prefill_profile():
    global _GLM_DSA_PREFILL_PROFILE
    _GLM_DSA_PREFILL_PROFILE = _new_profile()


def get_glm_dsa_prefill_profile(reset: bool = False):
    global _GLM_DSA_PREFILL_PROFILE
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    profile = {
        "stages": {
            stage: dict(values)
            for stage, values in _GLM_DSA_PREFILL_PROFILE["stages"].items()
        },
        "fast_prefill_hits": _GLM_DSA_PREFILL_PROFILE["fast_prefill_hits"],
        "fallback_reasons": dict(_GLM_DSA_PREFILL_PROFILE["fallback_reasons"]),
    }
    if reset:
        reset_glm_dsa_prefill_profile()
    return profile


def _record_stage(stage: str, seconds: float):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    values = _GLM_DSA_PREFILL_PROFILE["stages"][stage]
    values["seconds"] += seconds
    values["count"] += 1


def _record_fast_prefill_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["fast_prefill_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["fallback_reasons"][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info("GLM DSA fast sparse prefill enabled")
        else:
            _LOGGER.info("GLM DSA fast sparse prefill fallback: %s", reason)


def _warn_fast_prefill_large_topk(topk: int):
    global _WARNED_FAST_PREFILL_LARGE_TOPK
    if _WARNED_FAST_PREFILL_LARGE_TOPK or topk < _FAST_PREFILL_LARGE_TOPK_WARNING:
        return
    _WARNED_FAST_PREFILL_LARGE_TOPK = True
    _LOGGER.warning(
        "GLM DSA exact sparse prefill is enabled with topk=%s. "
        "This path is an opt-in diagnostic path and was slower than fallback "
        "in GLM-5.2 4k profiling at index_topk=2048.",
        topk,
    )


def _collect_arrays(value, arrays):
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_arrays(item, arrays)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_arrays(item, arrays)


def _eval_profile_value(value):
    arrays = []
    _collect_arrays(value, arrays)
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _profile_stage(stage: str, fn):
    if not _prefill_profile_enabled():
        return fn()
    start = time.perf_counter()
    value = fn()
    _eval_profile_value(value)
    _record_stage(stage, time.perf_counter() - start)
    return value


def _scalar_int(value):
    if isinstance(value, int):
        return value
    if hasattr(value, "shape"):
        if value.shape != ():
            return None
        try:
            return int(value.item())
        except Exception:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gather_sequence_by_flat_index(x: mx.array, indices: mx.array) -> mx.array:
    B, H, S, D = x.shape
    iB, iH, L, K = indices.shape
    if iB != B:
        raise ValueError("top-k batch dimension does not match K/V batch dimension")
    if iH == 1 and H != 1:
        indices = mx.broadcast_to(indices, (B, H, L, K))
    elif iH != H:
        raise ValueError("top-k head dimension does not match K/V head dimension")
    offsets = mx.arange(B * H).reshape(B, H, 1, 1) * S
    flat_indices = indices + offsets
    return mx.take(x.reshape(B * H * S, D), flat_indices, axis=0)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    self.indexer_types = [
                        {"F": "full", "S": "shared"}[c] for c in pattern
                    ]
                else:
                    self.indexer_types = list(pattern)
            else:
                freq = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

    def _fast_prefill_decision(
        self,
        *,
        B: int,
        L: int,
        cache: Optional[Any],
        topk_indices: Optional[mx.array],
        kv_latent: mx.array,
        k_pe: mx.array,
    ):
        if not _fast_prefill_enabled():
            return False, "disabled"
        if L <= 1:
            return False, "decode"
        if topk_indices is None:
            return False, "no_topk_indices"
        if cache is None:
            return False, "no_mla_cache"
        try:
            kv_cache = cache[0]
        except (IndexError, TypeError):
            return False, "no_mla_cache"
        if kv_cache is None:
            return False, "no_mla_cache"
        if B != 1:
            return False, "batch_size_not_one"
        if not isinstance(kv_cache, (GlmMlaKVCache, QuantizedGlmMlaKVCache)):
            return False, f"unsupported_cache:{type(kv_cache).__name__}"
        if len(topk_indices.shape) != 4:
            return False, "topk_rank"
        if topk_indices.shape[0] != B or topk_indices.shape[2] != L:
            return False, "topk_shape"
        if topk_indices.shape[1] not in (1, self.num_heads):
            return False, "topk_heads"
        K = topk_indices.shape[-1]
        if K <= 0:
            return False, "empty_topk"
        if K > kv_latent.shape[2]:
            return False, "topk_exceeds_context"
        if kv_latent.shape[1] != 1 or k_pe.shape[1] != 1:
            return False, "unsupported_kv_heads"
        offset = _scalar_int(kv_cache.offset)
        if offset is None:
            return False, "non_scalar_offset"
        if offset + 1 < K:
            return False, "causal_prefix_shorter_than_topk"
        _warn_fast_prefill_large_topk(K)
        return True, "fast"

    def _fast_sparse_prefill_attention(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_latent: mx.array,
        k_pe: mx.array,
        topk_indices: mx.array,
    ):
        _, _, L, _ = q_nope.shape
        query_chunk = _fast_prefill_query_chunk_size(L)

        k, v = _profile_stage(
            "latent_kv_projection",
            lambda: (
                self.embed_q(kv_latent, transpose=False),
                self.unembed_out(kv_latent),
            ),
        )

        outputs = []
        for start in range(0, L, query_chunk):
            stop = min(start + query_chunk, L)
            chunk_topk = topk_indices[:, :, start:stop, :]

            k_selected, v_selected, k_pe_selected = _profile_stage(
                "sparse_gather",
                lambda chunk_topk=chunk_topk: (
                    _gather_sequence_by_flat_index(k, chunk_topk),
                    _gather_sequence_by_flat_index(v, chunk_topk),
                    _gather_sequence_by_flat_index(k_pe, chunk_topk),
                ),
            )

            q_nope_chunk = q_nope[:, :, start:stop, :]
            q_pe_chunk = q_pe[:, :, start:stop, :]
            outputs.append(
                _profile_stage(
                    "attention",
                    lambda q_nope_chunk=q_nope_chunk,
                    q_pe_chunk=q_pe_chunk,
                    k_selected=k_selected,
                    v_selected=v_selected,
                    k_pe_selected=k_pe_selected: self._fast_sparse_attention_chunk(
                        q_nope_chunk,
                        q_pe_chunk,
                        k_selected,
                        v_selected,
                        k_pe_selected,
                    ),
                )
            )

        return mx.concatenate(outputs, axis=2)

    def _fast_sparse_attention_chunk(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        k_selected: mx.array,
        v_selected: mx.array,
        k_pe_selected: mx.array,
    ):
        B, H, L, D = q_nope.shape
        K = k_selected.shape[-2]
        V = v_selected.shape[-1]
        pe_scores = mx.sum(
            (q_pe * self.scale)[..., None, :] * k_pe_selected,
            axis=-1,
        )
        output = mx.fast.scaled_dot_product_attention(
            q_nope.reshape(B * H * L, 1, 1, D),
            k_selected.reshape(B * H * L, 1, K, D),
            v_selected.reshape(B * H * L, 1, K, V),
            scale=self.scale,
            mask=pe_scores.reshape(B * H * L, 1, 1, K),
        )
        return output.reshape(B, H, L, V)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        B, L, D = x.shape
        profile_total = _prefill_profile_enabled() and L > 1
        total_start = time.perf_counter() if profile_total else None

        def project_q():
            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)
            q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(
                0, 2, 1, 3
            )
            q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
            return qr, q_nope, q_pe

        qr, q_nope, q_pe = _profile_stage("q_projection", project_q)

        offset = cache[0].offset if cache is not None else 0

        def update_kv_cache():
            compressed_kv = self.kv_a_proj_with_mqa(x)
            compressed_kv, k_pe = mx.split(
                compressed_kv, [self.kv_lora_rank], axis=-1
            )
            k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(
                0, 2, 1, 3
            )
            kv_latent = self.kv_a_layernorm(compressed_kv)

            q_pe_rope = self.rope(q_pe, offset)
            k_pe = self.rope(k_pe, offset)

            kv_latent = mx.expand_dims(kv_latent, axis=1)

            if cache is not None:
                return cache[0].update_and_fetch(kv_latent, k_pe), q_pe_rope
            return (kv_latent, k_pe), q_pe_rope

        (kv_latent, k_pe), q_pe = _profile_stage("kv_cache_update", update_kv_cache)

        if cache is not None:
            if hasattr(cache[0], "dequantize_keys"):
                kv_latent = _profile_stage(
                    "latent_kv_dequantization",
                    lambda: cache[0].dequantize_keys(kv_latent),
                )
        else:
            cache = [None] * 2

        if self.indexer is not None:
            topk_indices = _profile_stage(
                "dsa_indexer_topk",
                lambda: self.indexer(x, qr, mask, cache=cache[1]),
            )
        else:
            topk_indices = prev_topk_indices

        fast_sparse_prefill = False
        if topk_indices is not None:
            if L == 1:
                _record_fast_prefill_decision(False, "decode")
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                fast_sparse_prefill, reason = self._fast_prefill_decision(
                    B=B,
                    L=L,
                    cache=cache,
                    topk_indices=topk_indices,
                    kv_latent=kv_latent,
                    k_pe=k_pe,
                )
                _record_fast_prefill_decision(fast_sparse_prefill, reason)
                if not fast_sparse_prefill:
                    shape = list(topk_indices.shape)
                    shape[-1] = kv_latent.shape[2]
                    sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(
                        sparse_mask, topk_indices, mx.array(True), axis=-1
                    )
                    if mask is not None:
                        sparse_mask = sparse_mask & mask
                    mask = sparse_mask
        elif L > 1:
            _record_fast_prefill_decision(False, "no_topk_indices")

        # Ensure the indexer cache is evaluated even if the topk_indices are unused
        # to keep the graph from getting too large
        if self.indexer is not None and cache is not None and cache[0] is not None:
            if isinstance(cache[0].keys, (tuple, list)):
                cache[0].keys = tuple(
                    mx.depends(k, (cache[1].keys, cache[1].values))
                    for k in cache[0].keys
                )
            else:
                cache[0].keys = mx.depends(
                    cache[0].keys, (cache[1].keys, cache[1].values)
                )

        if fast_sparse_prefill:
            output = self._fast_sparse_prefill_attention(
                q_nope,
                q_pe,
                kv_latent,
                k_pe,
                topk_indices,
            )
        else:
            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )

            if L == 1:
                q_nope = self.embed_q(q_nope)
                k = v = kv_latent
            else:
                k, v = _profile_stage(
                    "latent_kv_projection",
                    lambda: (
                        self.embed_q(kv_latent, transpose=False),
                        self.unembed_out(kv_latent),
                    ),
                )

            output = _profile_stage(
                "attention",
                lambda: scaled_dot_product_attention(
                    q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
                ),
            )
            if L == 1:
                output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)
        if profile_total:
            _eval_profile_value(output)
            _record_stage("total_prefill", time.perf_counter() - total_start)
        return output, topk_indices


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, prev_topk_indices
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)

    def make_cache(self):
        # Shared layers run no indexer, so they get no indexer KVCache.
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(GlmMlaKVCache()))
            else:
                caches.append(CacheList(GlmMlaKVCache(), KVCache()))
        return caches
