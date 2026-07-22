# Copyright © 2025 Apple Inc.

import logging
import os
import time
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import (
    BatchGlmMlaKVCache,
    BatchQuantizedGlmMlaKVCache,
    CacheList,
    GlmMlaKVCache,
    KVCache,
    QuantizedGlmMlaKVCache,
)
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from .deepseek_v32 import Model as DSV32Model
from .mla import QuantizedMultiLinear


GLM_DSA_FAST_PREFILL_ENV = "MLX_LM_GLM_DSA_FAST_PREFILL"
GLM_DSA_FAST_PREFILL_DEBUG_ENV = "MLX_LM_GLM_DSA_FAST_PREFILL_DEBUG"
GLM_DSA_FAST_PREFILL_QUERY_CHUNK_ENV = "MLX_LM_GLM_DSA_FAST_PREFILL_QUERY_CHUNK"
GLM_DSA_FAST_PREFILL_KEY_BLOCK_ENV = "MLX_LM_GLM_DSA_FAST_PREFILL_KEY_BLOCK"
GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV = (
    "MLX_LM_GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT"
)
GLM_DSA_NATIVE_SPARSE_PREFILL_ENV = "MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL"
GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT_ENV = (
    "MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT"
)
GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV = (
    "MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV"
)
GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT_ENV = (
    "MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT"
)
GLM_DSA_SPARSE_MLA_TILE_ENV = "MLX_LM_GLM_DSA_SPARSE_MLA_TILE"
GLM_DSA_NATIVE_INDEXER_ENV = "MLX_LM_GLM_DSA_NATIVE_INDEXER"
GLM_DSA_NATIVE_DECODE_INDEXER_ENV = "MLX_LM_GLM_DSA_NATIVE_DECODE_INDEXER"
GLM_DSA_NATIVE_INDEXER_MAX_SCORE_BYTES_ENV = (
    "MLX_LM_GLM_DSA_NATIVE_INDEXER_MAX_SCORE_BYTES"
)
GLM_DSA_NATIVE_Q8_VUP_ENV = "MLX_LM_GLM_DSA_NATIVE_Q8_VUP"
GLM_DSA_NATIVE_Q4_VUP_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_VUP"
GLM_DSA_Q_A_DENSE_CACHE_ENV = "MLX_LM_GLM_DSA_Q_A_DENSE_CACHE"
GLM_DSA_NATIVE_Q4_QA_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QA"
GLM_DSA_NATIVE_Q4_QA_TILE_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QA_TILE"
GLM_DSA_NATIVE_Q4_QB_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QB"
GLM_DSA_NATIVE_Q4_QB_TILE_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QB_TILE"
GLM_DSA_PREFILL_PROFILE_ENV = "MLX_LM_GLM_DSA_PREFILL_PROFILE"
GLM_DSA_PREFILL_PROFILE_ISOLATE_ENV = "MLX_LM_GLM_DSA_PREFILL_PROFILE_ISOLATE"
GLM_DSA_DECODE_PROFILE_ENV = "MLX_LM_GLM_DSA_DECODE_PROFILE"
GLM_DSA_DECODE_PROFILE_ISOLATE_ENV = "MLX_LM_GLM_DSA_DECODE_PROFILE_ISOLATE"
GLM_DSA_MTP_ENV = "MLX_LM_GLM_DSA_MTP"

_PROFILE_STAGES = (
    "input_layernorm",
    "q_projection",
    "q_a_projection",
    "q_a_dense_cache_dequantization",
    "q_a_dense_projection",
    "native_q4_qa_projection",
    "q_a_layernorm",
    "q_b_projection",
    "native_q4_qb_projection",
    "kv_cache_update",
    "dsa_indexer_topk",
    "native_indexer_scores",
    "native_indexer_topk",
    "latent_kv_dequantization",
    "latent_kv_projection",
    "native_sparse_kv_dequantization",
    "native_q8_vup",
    "native_q4_vup",
    "sparse_gather",
    "attention",
    "native_sparse_attention",
    "o_projection",
    "post_attention_layernorm",
    "mlp",
    "total_prefill",
)
_DECODE_PROFILE_STAGES = tuple(
    stage for stage in _PROFILE_STAGES if stage != "total_prefill"
) + (
    "total_decode_attention",
    "total_decode_layer",
    "total_decode_model",
)
_DEFAULT_FAST_PREFILL_QUERY_CHUNK = 16
_DEFAULT_FAST_PREFILL_KEY_BLOCK = 8192
_DEFAULT_SPARSE_PREFILL_MIN_CONTEXT = 131072
_DEFAULT_NATIVE_SPARSE_PREFILL_MIN_CONTEXT = 6144
_DEFAULT_NATIVE_INDEXER_MAX_SCORE_BYTES = 256 * 1024 * 1024
_MAX_QUANTIZED_SPARSE_VERIFY_TOKENS = 8
_FAST_PREFILL_LARGE_TOPK_WARNING = 1024
_LOGGER = logging.getLogger(__name__)
_GLM_DSA_PREFILL_PROFILE = None
_GLM_DSA_DECODE_PROFILE = None
_GLM_DSA_PROFILE_SCOPE = ContextVar("glm_dsa_profile_scope", default=None)
_GLM_DSA_MTP_VERIFY_SCOPE = ContextVar("glm_dsa_mtp_verify_scope", default=False)
_WARNED_FAST_PREFILL_LARGE_TOPK = False
_NATIVE_SPARSE_MLA_LOOKUP_DONE = False
_NATIVE_SPARSE_MLA_KERNEL = None
_NATIVE_SPARSE_MLA_SOURCE = None
_NATIVE_SPARSE_MLA_IMPORT_ERROR = None
_NATIVE_Q8_VUP_LOOKUP_DONE = False
_NATIVE_Q8_VUP_KERNEL = None
_NATIVE_Q8_VUP_SOURCE = None
_NATIVE_Q8_VUP_IMPORT_ERROR = None
_NATIVE_Q4_VUP_LOOKUP_DONE = False
_NATIVE_Q4_VUP_KERNEL = None
_NATIVE_Q4_VUP_SOURCE = None
_NATIVE_Q4_VUP_IMPORT_ERROR = None
_NATIVE_Q4_QA_LOOKUP_DONE = False
_NATIVE_Q4_QA_KERNEL = None
_NATIVE_Q4_QA_SOURCE = None
_NATIVE_Q4_QA_IMPORT_ERROR = None
_NATIVE_Q4_QB_LOOKUP_DONE = False
_NATIVE_Q4_QB_KERNEL = None
_NATIVE_Q4_QB_SOURCE = None
_NATIVE_Q4_QB_IMPORT_ERROR = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _fast_prefill_enabled() -> bool:
    return _env_flag(GLM_DSA_FAST_PREFILL_ENV, True)


def _native_sparse_prefill_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_SPARSE_PREFILL_ENV, True)


def _native_sparse_prefill_quantized_kv_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV, False)


def _native_indexer_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_INDEXER_ENV, True)


def _native_decode_indexer_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_DECODE_INDEXER_ENV, False)


def _native_q8_vup_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q8_VUP_ENV, False)


def _native_q4_vup_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q4_VUP_ENV, False)


def _q_a_dense_cache_enabled() -> bool:
    return _env_flag(GLM_DSA_Q_A_DENSE_CACHE_ENV, False)


def _native_q4_qa_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q4_QA_ENV, False)


def _native_q4_qb_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q4_QB_ENV, False)


def _prefill_profile_enabled() -> bool:
    return _env_flag(GLM_DSA_PREFILL_PROFILE_ENV, False)


def _prefill_profile_isolate_enabled() -> bool:
    return _env_flag(GLM_DSA_PREFILL_PROFILE_ISOLATE_ENV, False)


def _decode_profile_enabled() -> bool:
    return _env_flag(GLM_DSA_DECODE_PROFILE_ENV, False)


def _decode_profile_isolate_enabled() -> bool:
    return _env_flag(GLM_DSA_DECODE_PROFILE_ISOLATE_ENV, False)


def _mtp_enabled() -> bool:
    return _env_flag(GLM_DSA_MTP_ENV, False)


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


def _fast_prefill_key_block_size(sequence_length: int, topk: int) -> int:
    raw_value = os.environ.get(GLM_DSA_FAST_PREFILL_KEY_BLOCK_ENV)
    default = max(topk, _DEFAULT_FAST_PREFILL_KEY_BLOCK)
    if raw_value is not None:
        try:
            default = int(raw_value)
        except ValueError:
            pass
    return max(topk, min(sequence_length, default))


def _sparse_prefill_min_context_length() -> int:
    raw_value = os.environ.get(GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV)
    if raw_value is not None:
        try:
            return max(0, int(raw_value))
        except ValueError:
            pass
    return _DEFAULT_SPARSE_PREFILL_MIN_CONTEXT


def _native_sparse_prefill_min_context_length() -> int:
    raw_value = os.environ.get(GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT_ENV)
    if raw_value is not None:
        try:
            return max(0, int(raw_value))
        except ValueError:
            pass
    return _DEFAULT_NATIVE_SPARSE_PREFILL_MIN_CONTEXT


def _native_sparse_prefill_quantized_kv_max_context_length() -> int:
    raw_value = os.environ.get(
        GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT_ENV
    )
    if raw_value is not None:
        try:
            return max(0, int(raw_value))
        except ValueError:
            pass
    return 65536


def _native_indexer_max_score_bytes() -> int:
    raw_value = os.environ.get(GLM_DSA_NATIVE_INDEXER_MAX_SCORE_BYTES_ENV)
    if raw_value is not None:
        try:
            return max(4, int(raw_value))
        except ValueError:
            pass
    return _DEFAULT_NATIVE_INDEXER_MAX_SCORE_BYTES


def _native_indexer_query_chunk_size(query_length: int, key_length: int) -> int:
    if query_length == 1:
        return 1
    if _native_indexer_max_score_bytes() < 4 * key_length * 64:
        return 0
    max_rows = max(1, _native_indexer_max_score_bytes() // (4 * key_length))
    chunk = min(query_length, max_rows)
    if chunk >= 64 and chunk < query_length:
        chunk = (chunk // 64) * 64
    return max(1, chunk)


def _sparse_prefill_min_effective_context_length() -> int:
    # Generation prefill leaves the final prompt token for logits, so a nominal
    # N-token prompt can expose at most N-1 tokens to the attention call.
    return max(0, _sparse_prefill_min_context_length() - 1)


def _sparse_prefill_context_ready(context_length: int) -> bool:
    return context_length >= _sparse_prefill_min_effective_context_length()


def _new_profile():
    return {
        "stages": {
            stage: {"seconds": 0.0, "count": 0} for stage in _PROFILE_STAGES
        },
        "fast_prefill_hits": 0,
        "fallback_reasons": Counter(),
        "native_sparse_prefill_hits": 0,
        "native_sparse_prefill_fallback_reasons": Counter(),
        "native_indexer_hits": 0,
        "native_indexer_fallback_reasons": Counter(),
        "native_q8_vup_hits": 0,
        "native_q8_vup_fallback_reasons": Counter(),
        "native_q4_vup_hits": 0,
        "native_q4_vup_fallback_reasons": Counter(),
        "q_a_dense_cache_hits": 0,
        "q_a_dense_cache_builds": 0,
        "q_a_dense_cache_fallback_reasons": Counter(),
        "native_q4_qa_hits": 0,
        "native_q4_qa_fallback_reasons": Counter(),
        "native_q4_qb_hits": 0,
        "native_q4_qb_fallback_reasons": Counter(),
    }


def _new_decode_profile():
    return {
        "stages": {
            stage: {"seconds": 0.0, "count": 0}
            for stage in _DECODE_PROFILE_STAGES
        },
    }


def reset_glm_dsa_prefill_profile():
    global _GLM_DSA_PREFILL_PROFILE
    _GLM_DSA_PREFILL_PROFILE = _new_profile()


def reset_glm_dsa_decode_profile():
    global _GLM_DSA_DECODE_PROFILE
    _GLM_DSA_DECODE_PROFILE = _new_decode_profile()


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
        "native_sparse_prefill_hits": _GLM_DSA_PREFILL_PROFILE[
            "native_sparse_prefill_hits"
        ],
        "native_sparse_prefill_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["native_sparse_prefill_fallback_reasons"]
        ),
        "native_indexer_hits": _GLM_DSA_PREFILL_PROFILE["native_indexer_hits"],
        "native_indexer_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["native_indexer_fallback_reasons"]
        ),
        "native_q8_vup_hits": _GLM_DSA_PREFILL_PROFILE["native_q8_vup_hits"],
        "native_q8_vup_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["native_q8_vup_fallback_reasons"]
        ),
        "native_q4_vup_hits": _GLM_DSA_PREFILL_PROFILE["native_q4_vup_hits"],
        "native_q4_vup_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["native_q4_vup_fallback_reasons"]
        ),
        "q_a_dense_cache_hits": _GLM_DSA_PREFILL_PROFILE["q_a_dense_cache_hits"],
        "q_a_dense_cache_builds": _GLM_DSA_PREFILL_PROFILE[
            "q_a_dense_cache_builds"
        ],
        "q_a_dense_cache_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["q_a_dense_cache_fallback_reasons"]
        ),
        "native_q4_qa_hits": _GLM_DSA_PREFILL_PROFILE["native_q4_qa_hits"],
        "native_q4_qa_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["native_q4_qa_fallback_reasons"]
        ),
        "native_q4_qb_hits": _GLM_DSA_PREFILL_PROFILE["native_q4_qb_hits"],
        "native_q4_qb_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["native_q4_qb_fallback_reasons"]
        ),
    }
    if reset:
        reset_glm_dsa_prefill_profile()
    return profile


def get_glm_dsa_decode_profile(reset: bool = False):
    global _GLM_DSA_DECODE_PROFILE
    if _GLM_DSA_DECODE_PROFILE is None:
        reset_glm_dsa_decode_profile()
    profile = {
        "stages": {
            stage: dict(values)
            for stage, values in _GLM_DSA_DECODE_PROFILE["stages"].items()
        },
    }
    if reset:
        reset_glm_dsa_decode_profile()
    return profile


def _record_stage(stage: str, seconds: float):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    values = _GLM_DSA_PREFILL_PROFILE["stages"][stage]
    values["seconds"] += seconds
    values["count"] += 1


def _record_decode_stage(stage: str, seconds: float):
    if _GLM_DSA_DECODE_PROFILE is None:
        reset_glm_dsa_decode_profile()
    values = _GLM_DSA_DECODE_PROFILE["stages"][stage]
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


def _record_native_sparse_prefill_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["native_sparse_prefill_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["native_sparse_prefill_fallback_reasons"][
            reason
        ] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info(
                "GLM DSA native sparse MLA prefill enabled: source=%s",
                _NATIVE_SPARSE_MLA_SOURCE or "unknown",
            )
        else:
            _LOGGER.info(
                "GLM DSA native sparse MLA prefill fallback: %s", reason
            )


def _record_native_indexer_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["native_indexer_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["native_indexer_fallback_reasons"][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info("GLM DSA native indexer enabled")
        else:
            _LOGGER.info("GLM DSA native indexer fallback: %s", reason)


def _record_native_q8_vup_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["native_q8_vup_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["native_q8_vup_fallback_reasons"][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info(
                "GLM DSA native q8 V-up projection enabled: source=%s",
                _NATIVE_Q8_VUP_SOURCE or "unknown",
            )
        else:
            _LOGGER.info("GLM DSA native q8 V-up fallback: %s", reason)


def _record_native_q4_vup_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["native_q4_vup_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["native_q4_vup_fallback_reasons"][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info(
                "GLM DSA native q4 V-up projection enabled: source=%s",
                _NATIVE_Q4_VUP_SOURCE or "unknown",
            )
        else:
            _LOGGER.info("GLM DSA native q4 V-up fallback: %s", reason)


def _record_native_q4_qb_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["native_q4_qb_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["native_q4_qb_fallback_reasons"][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info(
                "GLM DSA native q4 q_b projection enabled: source=%s",
                _NATIVE_Q4_QB_SOURCE or "unknown",
            )
        else:
            _LOGGER.info("GLM DSA native q4 q_b fallback: %s", reason)


def _record_q_a_dense_cache_decision(
    used: bool,
    reason: str,
    *,
    built: bool = False,
):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["q_a_dense_cache_hits"] += 1
        if built:
            _GLM_DSA_PREFILL_PROFILE["q_a_dense_cache_builds"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["q_a_dense_cache_fallback_reasons"][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info(
                "GLM DSA q_a dense cache enabled%s",
                " with build" if built else "",
            )
        else:
            _LOGGER.info("GLM DSA q_a dense cache fallback: %s", reason)


def _record_native_q4_qa_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["native_q4_qa_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE["native_q4_qa_fallback_reasons"][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info(
                "GLM DSA native q4 q_a projection enabled: source=%s",
                _NATIVE_Q4_QA_SOURCE or "unknown",
            )
        else:
            _LOGGER.info("GLM DSA native q4 q_a fallback: %s", reason)


def _native_sparse_mla_kernel():
    global _NATIVE_SPARSE_MLA_LOOKUP_DONE
    global _NATIVE_SPARSE_MLA_KERNEL
    global _NATIVE_SPARSE_MLA_SOURCE
    global _NATIVE_SPARSE_MLA_IMPORT_ERROR
    if _NATIVE_SPARSE_MLA_LOOKUP_DONE:
        return _NATIVE_SPARSE_MLA_KERNEL

    _NATIVE_SPARSE_MLA_LOOKUP_DONE = True
    _NATIVE_SPARSE_MLA_KERNEL = None
    _NATIVE_SPARSE_MLA_SOURCE = None
    _NATIVE_SPARSE_MLA_IMPORT_ERROR = None

    for module_name in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        try:
            fast = __import__(module_name, fromlist=["fast"]).fast
            has_symbol = getattr(fast, "has_symbol", None)
            if (
                has_symbol is not None
                and has_symbol("glm_dsa_sparse_mla_attention")
                and hasattr(fast, "glm_dsa_sparse_mla_attention")
            ):
                _NATIVE_SPARSE_MLA_KERNEL = fast.glm_dsa_sparse_mla_attention
                _NATIVE_SPARSE_MLA_SOURCE = module_name
                return _NATIVE_SPARSE_MLA_KERNEL
            if _NATIVE_SPARSE_MLA_IMPORT_ERROR is None and hasattr(
                fast, "import_error"
            ):
                _NATIVE_SPARSE_MLA_IMPORT_ERROR = fast.import_error()
        except Exception as exc:
            if _NATIVE_SPARSE_MLA_IMPORT_ERROR is None:
                _NATIVE_SPARSE_MLA_IMPORT_ERROR = exc

    if hasattr(mx.fast, "glm_dsa_sparse_mla_attention"):
        _NATIVE_SPARSE_MLA_KERNEL = mx.fast.glm_dsa_sparse_mla_attention
        _NATIVE_SPARSE_MLA_SOURCE = "mlx.core.fast"

    return _NATIVE_SPARSE_MLA_KERNEL


def _native_q8_vup_kernel():
    global _NATIVE_Q8_VUP_LOOKUP_DONE
    global _NATIVE_Q8_VUP_KERNEL
    global _NATIVE_Q8_VUP_SOURCE
    global _NATIVE_Q8_VUP_IMPORT_ERROR
    if _NATIVE_Q8_VUP_LOOKUP_DONE:
        return _NATIVE_Q8_VUP_KERNEL

    _NATIVE_Q8_VUP_LOOKUP_DONE = True
    _NATIVE_Q8_VUP_KERNEL = None
    _NATIVE_Q8_VUP_SOURCE = None
    _NATIVE_Q8_VUP_IMPORT_ERROR = None

    for module_name in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        try:
            fast = __import__(module_name, fromlist=["fast"]).fast
            has_symbol = getattr(fast, "has_symbol", None)
            if (
                has_symbol is not None
                and has_symbol("glm_dsa_q8_vup_flat")
                and hasattr(fast, "glm_dsa_q8_vup_flat")
            ):
                _NATIVE_Q8_VUP_KERNEL = fast.glm_dsa_q8_vup_flat
                _NATIVE_Q8_VUP_SOURCE = module_name
                return _NATIVE_Q8_VUP_KERNEL
            if _NATIVE_Q8_VUP_IMPORT_ERROR is None and hasattr(
                fast, "import_error"
            ):
                _NATIVE_Q8_VUP_IMPORT_ERROR = fast.import_error()
        except Exception as exc:
            if _NATIVE_Q8_VUP_IMPORT_ERROR is None:
                _NATIVE_Q8_VUP_IMPORT_ERROR = exc

    if hasattr(mx.fast, "glm_dsa_q8_vup_flat"):
        _NATIVE_Q8_VUP_KERNEL = mx.fast.glm_dsa_q8_vup_flat
        _NATIVE_Q8_VUP_SOURCE = "mlx.core.fast"

    return _NATIVE_Q8_VUP_KERNEL


def _native_q4_vup_kernel():
    global _NATIVE_Q4_VUP_LOOKUP_DONE
    global _NATIVE_Q4_VUP_KERNEL
    global _NATIVE_Q4_VUP_SOURCE
    global _NATIVE_Q4_VUP_IMPORT_ERROR
    if _NATIVE_Q4_VUP_LOOKUP_DONE:
        return _NATIVE_Q4_VUP_KERNEL

    _NATIVE_Q4_VUP_LOOKUP_DONE = True
    _NATIVE_Q4_VUP_KERNEL = None
    _NATIVE_Q4_VUP_SOURCE = None
    _NATIVE_Q4_VUP_IMPORT_ERROR = None

    for module_name in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        try:
            fast = __import__(module_name, fromlist=["fast"]).fast
            has_symbol = getattr(fast, "has_symbol", None)
            if (
                has_symbol is not None
                and has_symbol("glm_dsa_q4_vup_flat")
                and hasattr(fast, "glm_dsa_q4_vup_flat")
            ):
                _NATIVE_Q4_VUP_KERNEL = fast.glm_dsa_q4_vup_flat
                _NATIVE_Q4_VUP_SOURCE = module_name
                return _NATIVE_Q4_VUP_KERNEL
            if _NATIVE_Q4_VUP_IMPORT_ERROR is None and hasattr(
                fast, "import_error"
            ):
                _NATIVE_Q4_VUP_IMPORT_ERROR = fast.import_error()
        except Exception as exc:
            if _NATIVE_Q4_VUP_IMPORT_ERROR is None:
                _NATIVE_Q4_VUP_IMPORT_ERROR = exc

    if hasattr(mx.fast, "glm_dsa_q4_vup_flat"):
        _NATIVE_Q4_VUP_KERNEL = mx.fast.glm_dsa_q4_vup_flat
        _NATIVE_Q4_VUP_SOURCE = "mlx.core.fast"

    return _NATIVE_Q4_VUP_KERNEL


def _native_q4_qb_kernel():
    global _NATIVE_Q4_QB_LOOKUP_DONE
    global _NATIVE_Q4_QB_KERNEL
    global _NATIVE_Q4_QB_SOURCE
    global _NATIVE_Q4_QB_IMPORT_ERROR
    if _NATIVE_Q4_QB_LOOKUP_DONE:
        return _NATIVE_Q4_QB_KERNEL

    _NATIVE_Q4_QB_LOOKUP_DONE = True
    _NATIVE_Q4_QB_KERNEL = None
    _NATIVE_Q4_QB_SOURCE = None
    _NATIVE_Q4_QB_IMPORT_ERROR = None

    for module_name in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        try:
            fast = __import__(module_name, fromlist=["fast"]).fast
            has_symbol = getattr(fast, "has_symbol", None)
            if (
                has_symbol is not None
                and has_symbol("glm_dsa_q4_qb_proj_flat")
                and hasattr(fast, "glm_dsa_q4_qb_proj_flat")
            ):
                _NATIVE_Q4_QB_KERNEL = fast.glm_dsa_q4_qb_proj_flat
                _NATIVE_Q4_QB_SOURCE = module_name
                return _NATIVE_Q4_QB_KERNEL
            if _NATIVE_Q4_QB_IMPORT_ERROR is None and hasattr(
                fast, "import_error"
            ):
                _NATIVE_Q4_QB_IMPORT_ERROR = fast.import_error()
        except Exception as exc:
            if _NATIVE_Q4_QB_IMPORT_ERROR is None:
                _NATIVE_Q4_QB_IMPORT_ERROR = exc

    if hasattr(mx.fast, "glm_dsa_q4_qb_proj_flat"):
        _NATIVE_Q4_QB_KERNEL = mx.fast.glm_dsa_q4_qb_proj_flat
        _NATIVE_Q4_QB_SOURCE = "mlx.core.fast"

    return _NATIVE_Q4_QB_KERNEL


def _native_q4_qa_kernel():
    global _NATIVE_Q4_QA_LOOKUP_DONE
    global _NATIVE_Q4_QA_KERNEL
    global _NATIVE_Q4_QA_SOURCE
    global _NATIVE_Q4_QA_IMPORT_ERROR
    if _NATIVE_Q4_QA_LOOKUP_DONE:
        return _NATIVE_Q4_QA_KERNEL

    _NATIVE_Q4_QA_LOOKUP_DONE = True
    _NATIVE_Q4_QA_KERNEL = None
    _NATIVE_Q4_QA_SOURCE = None
    _NATIVE_Q4_QA_IMPORT_ERROR = None

    for module_name in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        try:
            fast = __import__(module_name, fromlist=["fast"]).fast
            has_symbol = getattr(fast, "has_symbol", None)
            if (
                has_symbol is not None
                and has_symbol("glm_dsa_q4_qa_proj_flat")
                and hasattr(fast, "glm_dsa_q4_qa_proj_flat")
            ):
                _NATIVE_Q4_QA_KERNEL = fast.glm_dsa_q4_qa_proj_flat
                _NATIVE_Q4_QA_SOURCE = module_name
                return _NATIVE_Q4_QA_KERNEL
            if _NATIVE_Q4_QA_IMPORT_ERROR is None and hasattr(
                fast, "import_error"
            ):
                _NATIVE_Q4_QA_IMPORT_ERROR = fast.import_error()
        except Exception as exc:
            if _NATIVE_Q4_QA_IMPORT_ERROR is None:
                _NATIVE_Q4_QA_IMPORT_ERROR = exc

    if hasattr(mx.fast, "glm_dsa_q4_qa_proj_flat"):
        _NATIVE_Q4_QA_KERNEL = mx.fast.glm_dsa_q4_qa_proj_flat
        _NATIVE_Q4_QA_SOURCE = "mlx.core.fast"

    return _NATIVE_Q4_QA_KERNEL


def _native_indexer_fast_module():
    try:
        from mlx_lm.custom_kernels.glm_moe_dsa import fast
    except Exception as exc:
        return None, None, exc
    source = (
        "mlx_lm.custom_kernels.glm_moe_dsa"
        if fast.is_native_available()
        else "mlx.core.fast"
    )
    return fast, source, fast.import_error()


def _native_indexer_available():
    fast, _source, _error = _native_indexer_fast_module()
    if fast is None:
        return False
    return fast.has_symbol("dsa_indexer_scores_fp32") and fast.has_symbol(
        "dsa_topk_indices_fp32"
    )


def _native_decode_indexer_available():
    fast, _source, _error = _native_indexer_fast_module()
    if fast is None:
        return False
    return fast.has_symbol("dsa_indexer_scores_decode_fp32") and fast.has_symbol(
        "dsa_topk_indices_fp32"
    )


def _native_indexer_scores(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    *,
    scale: float,
    causal: bool,
    skip_causal_future_store: bool = False,
    causal_q_offset: int = -1,
):
    fast, _source, _error = _native_indexer_fast_module()
    if (
        fast is None
        or not fast.has_symbol("dsa_indexer_scores_fp32")
        or len(queries.shape) != 4
        or len(keys.shape) != 4
        or len(weights.shape) != 3
        or queries.shape[0] != keys.shape[0]
        or queries.shape[0] != weights.shape[0]
        or queries.shape[1] != 32
        or keys.shape[1] != 1
        or queries.shape[2] != weights.shape[1]
        or queries.shape[1] != weights.shape[2]
        or queries.shape[3] != 128
        or keys.shape[3] != 128
        or keys.shape[2] < 4096
        or queries.dtype != keys.dtype
        or queries.dtype not in (mx.float16, mx.bfloat16)
        or weights.dtype != mx.float32
    ):
        return None

    _B, _H, L, _D = queries.shape
    K = keys.shape[2]
    if L == 1:
        if not fast.has_symbol("dsa_indexer_scores_decode_fp32"):
            return None
        try:
            scores = fast.dsa_indexer_scores_decode_fp32(
                queries,
                keys,
                weights,
                scale,
                stream=mx.gpu,
            )
            return scores if scores.dtype == mx.float32 else None
        except Exception:
            return None

    q_pad = (-L) % 64
    k_pad = (-K) % 64
    if causal and causal_q_offset < 0 and (q_pad or k_pad):
        causal_q_offset = K - L

    q = queries
    k = keys
    w = weights
    if q_pad:
        q = mx.pad(q, [(0, 0), (0, 0), (0, q_pad), (0, 0)])
        w = mx.pad(w, [(0, 0), (0, q_pad), (0, 0)])
    if k_pad:
        k = mx.pad(k, [(0, 0), (0, 0), (0, k_pad), (0, 0)])

    try:
        scores = fast.dsa_indexer_scores_fp32(
            q,
            k,
            w,
            scale,
            causal=causal,
            unused_causal_prefix_topk=0,
            skip_causal_future_store=skip_causal_future_store,
            causal_q_offset=causal_q_offset,
            stream=mx.gpu,
        )
    except Exception:
        return None
    if scores.dtype != mx.float32:
        return None
    if q_pad or k_pad:
        scores = scores[:, :, :L, :K]
    return scores


def _native_indexer_topk_indices(
    scores: mx.array,
    topk: int,
    *,
    bucketed: bool,
    causal_valid_prefix: bool,
):
    fast, _source, _error = _native_indexer_fast_module()
    if (
        fast is None
        or not fast.has_symbol("dsa_topk_indices_fp32")
        or len(scores.shape) != 4
        or scores.shape[1] != 1
        or topk != 2048
        or scores.shape[-1] < topk
        or scores.dtype != mx.float32
    ):
        return None
    try:
        return fast.dsa_topk_indices_fp32(
            scores,
            topk,
            bucketed=bucketed,
            causal_valid_prefix=causal_valid_prefix,
            stream=mx.gpu,
        )
    except Exception:
        return None


def get_glm_dsa_native_sparse_prefill_status():
    kernel = _native_sparse_mla_kernel()
    return {
        "enabled": _native_sparse_prefill_enabled(),
        "available": kernel is not None,
        "source": _NATIVE_SPARSE_MLA_SOURCE,
        "import_error": (
            repr(_NATIVE_SPARSE_MLA_IMPORT_ERROR)
            if _NATIVE_SPARSE_MLA_IMPORT_ERROR is not None
            else None
        ),
        "min_context": _native_sparse_prefill_min_context_length(),
        "quantized_kv_enabled": _native_sparse_prefill_quantized_kv_enabled(),
        "quantized_kv_max_context": (
            _native_sparse_prefill_quantized_kv_max_context_length()
        ),
    }


def get_glm_dsa_native_indexer_status():
    fast, source, import_error = _native_indexer_fast_module()
    scores_available = fast is not None and fast.has_symbol(
        "dsa_indexer_scores_fp32"
    )
    decode_scores_available = fast is not None and fast.has_symbol(
        "dsa_indexer_scores_decode_fp32"
    )
    topk_available = fast is not None and fast.has_symbol(
        "dsa_topk_indices_fp32"
    )
    return {
        "enabled": _native_indexer_enabled(),
        "decode_enabled": _native_decode_indexer_enabled(),
        "available": scores_available and topk_available,
        "source": source if scores_available and topk_available else None,
        "import_error": repr(import_error) if import_error is not None else None,
        "scores_available": scores_available,
        "decode_scores_available": decode_scores_available,
        "topk_available": topk_available,
        "min_context": 4096,
        "score_dtype": "float32",
        "max_score_bytes": _native_indexer_max_score_bytes(),
    }


def get_glm_dsa_native_q8_vup_status():
    kernel = _native_q8_vup_kernel()
    return {
        "enabled": _native_q8_vup_enabled(),
        "available": kernel is not None,
        "source": _NATIVE_Q8_VUP_SOURCE,
        "import_error": (
            repr(_NATIVE_Q8_VUP_IMPORT_ERROR)
            if _NATIVE_Q8_VUP_IMPORT_ERROR is not None
            else None
        ),
    }


def get_glm_dsa_native_q4_vup_status():
    kernel = _native_q4_vup_kernel()
    return {
        "enabled": _native_q4_vup_enabled(),
        "available": kernel is not None,
        "source": _NATIVE_Q4_VUP_SOURCE,
        "import_error": (
            repr(_NATIVE_Q4_VUP_IMPORT_ERROR)
            if _NATIVE_Q4_VUP_IMPORT_ERROR is not None
            else None
        ),
    }


def get_glm_dsa_native_q4_qb_status():
    kernel = _native_q4_qb_kernel()
    return {
        "enabled": _native_q4_qb_enabled(),
        "available": kernel is not None,
        "source": _NATIVE_Q4_QB_SOURCE,
        "import_error": (
            repr(_NATIVE_Q4_QB_IMPORT_ERROR)
            if _NATIVE_Q4_QB_IMPORT_ERROR is not None
            else None
        ),
    }


def get_glm_dsa_native_q4_qa_status():
    kernel = _native_q4_qa_kernel()
    return {
        "enabled": _native_q4_qa_enabled(),
        "available": kernel is not None,
        "source": _NATIVE_Q4_QA_SOURCE,
        "import_error": (
            repr(_NATIVE_Q4_QA_IMPORT_ERROR)
            if _NATIVE_Q4_QA_IMPORT_ERROR is not None
            else None
        ),
    }


def get_glm_dsa_mtp_status():
    return {
        "enabled": _mtp_enabled(),
        "env": GLM_DSA_MTP_ENV,
    }


def _warn_fast_prefill_large_topk(topk: int):
    global _WARNED_FAST_PREFILL_LARGE_TOPK
    if _WARNED_FAST_PREFILL_LARGE_TOPK or topk < _FAST_PREFILL_LARGE_TOPK_WARNING:
        return
    _WARNED_FAST_PREFILL_LARGE_TOPK = True
    _LOGGER.warning(
        "GLM DSA exact sparse prefill is enabled with topk=%s. "
        "This memory-bounded path avoids full-context dense prefill tensors, "
        "but can be slower than the old dense fallback at short context.",
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


def _profile_stage(stage: str, fn, *, inputs=None):
    scope = _GLM_DSA_PROFILE_SCOPE.get()
    if scope == "decode":
        if not _decode_profile_enabled():
            return fn()
        isolate = _decode_profile_isolate_enabled()
        record_stage = _record_decode_stage
    else:
        if not _prefill_profile_enabled():
            return fn()
        isolate = _prefill_profile_isolate_enabled()
        record_stage = _record_stage

    if inputs is not None and isolate:
        _eval_profile_value(inputs)
    start = time.perf_counter()
    value = fn()
    _eval_profile_value(value)
    record_stage(stage, time.perf_counter() - start)
    return value


def _profile_scope_enabled(scope: str) -> bool:
    if scope == "decode":
        return _decode_profile_enabled()
    if scope == "prefill":
        return _prefill_profile_enabled()
    return False


def _set_profile_scope(scope: Optional[str]):
    if scope is None or not _profile_scope_enabled(scope):
        return None
    return _GLM_DSA_PROFILE_SCOPE.set(scope)


def set_glm_dsa_mtp_verify_scope():
    return _GLM_DSA_MTP_VERIFY_SCOPE.set(True)


def reset_glm_dsa_mtp_verify_scope(token):
    _GLM_DSA_MTP_VERIFY_SCOPE.reset(token)


def _scalar_int(value):
    if isinstance(value, int):
        return value
    if hasattr(value, "shape"):
        size = 1
        for dim in value.shape:
            size *= dim
        if value.shape != () and size != 1:
            return None
        try:
            return int(value.item())
        except Exception:
            try:
                return int(value.tolist()[0])
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
    if H == 1 and iH != 1:
        offsets = mx.arange(B).reshape(B, 1, 1, 1) * S
        offsets = mx.broadcast_to(offsets, (B, iH, 1, 1))
        flat_indices = indices + offsets
        return mx.take(x.reshape(B * S, D), flat_indices, axis=0)
    if iH == 1 and H != 1:
        indices = mx.broadcast_to(indices, (B, H, L, K))
    elif iH != H:
        raise ValueError("top-k head dimension does not match K/V head dimension")
    offsets = mx.arange(B * H).reshape(B, H, 1, 1) * S
    flat_indices = indices + offsets
    return mx.take(x.reshape(B * H * S, D), flat_indices, axis=0)


def _slice_attention_mask(
    mask: Optional[mx.array],
    query_start: int,
    query_stop: int,
    key_start: int,
    key_stop: int,
):
    if mask is None:
        return None
    if len(mask.shape) == 2:
        return mask[query_start:query_stop, key_start:key_stop]
    return mask[..., query_start:query_stop, key_start:key_stop]


def _normalize_attention_mask(
    mask: Optional[mx.array], query_length: int, key_length: int
):
    """Materialize the string causal mask used by the generic MLX API."""
    if not isinstance(mask, str):
        return mask
    if mask != "causal":
        raise ValueError(f"unsupported attention mask: {mask!r}")
    query_positions = mx.arange(key_length - query_length, key_length)
    key_positions = mx.arange(key_length)
    return query_positions[:, None] >= key_positions[None, :]


def _apply_attention_mask(scores: mx.array, mask: Optional[mx.array]):
    """Apply boolean hard masks or additive masks to attention-like scores."""
    if mask is None:
        return scores
    if isinstance(mask, str):
        mask = _normalize_attention_mask(mask, scores.shape[-2], scores.shape[-1])
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.array(-float("inf"), scores.dtype))
    if not mx.issubdtype(mask.dtype, mx.floating):
        raise ValueError("custom attention masks must be boolean or floating point")
    return scores + mask.astype(scores.dtype)


def _validate_attention_mask(
    mask: Optional[mx.array],
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
):
    if mask is None:
        return
    if isinstance(mask, str):
        raise ValueError("string attention masks must be normalized before validation")
    if mask.dtype != mx.bool_ and not mx.issubdtype(mask.dtype, mx.floating):
        raise ValueError("custom attention masks must be boolean or floating point")
    if len(mask.shape) not in (2, 4):
        raise ValueError("custom attention masks must have rank 2 or 4")
    if tuple(mask.shape[-2:]) != (query_length, key_length):
        raise ValueError(
            "custom attention mask query/key dimensions do not match attention"
        )
    if len(mask.shape) == 4:
        mask_batch, mask_heads = mask.shape[:2]
        if mask_batch not in (1, batch_size):
            raise ValueError("custom attention mask batch dimension is not broadcastable")
        if mask_heads not in (1, num_heads):
            raise ValueError("custom attention mask head dimension is not broadcastable")


def _attention_mask_valid_rows(mask: Optional[mx.array]):
    if mask is None:
        return None
    if mask.dtype == mx.bool_:
        return mx.any(mask, axis=-1, keepdims=True)
    return mx.any(mx.isfinite(mask), axis=-1, keepdims=True)


def _ensure_attention_mask_has_valid_key(
    mask: Optional[mx.array], valid_rows: Optional[mx.array]
):
    """Give fully-masked rows a dummy key; their outputs are zeroed later."""
    if mask is None or valid_rows is None:
        return mask
    first_key = mx.arange(mask.shape[-1]) == 0
    dummy_key = (~valid_rows) & first_key
    if mask.dtype == mx.bool_:
        return mask | dummy_key
    return mx.where(dummy_key, mx.array(0, dtype=mask.dtype), mask)


def _has_canonical_causal_cache(cache: Optional[Any]) -> bool:
    """Whether cache.make_mask produces an unpadded, unwindowed causal mask."""
    return cache is None or type(cache) in (GlmMlaKVCache, QuantizedGlmMlaKVCache)


def _gather_attention_mask(mask: Optional[mx.array], indices: mx.array):
    if mask is None:
        return None
    B, iH, L, K = indices.shape
    if len(mask.shape) == 2:
        mL, S = mask.shape
        if mL != L:
            raise ValueError("mask query dimension does not match top-k indices")
        offsets = mx.arange(L).reshape(1, 1, L, 1) * S
        return mx.take(mask.reshape(mL * S), indices + offsets, axis=0)
    if len(mask.shape) != 4:
        raise ValueError("unsupported attention mask rank for sparse gather")

    mB, mH, mL, S = mask.shape
    if mB == 1 and B != 1:
        mask = mx.broadcast_to(mask, (B, mH, mL, S))
        mB = B
    if mB != B or mL != L:
        raise ValueError("mask shape does not match top-k indices")
    if mH == 1 and iH != 1:
        offsets = (mx.arange(B).reshape(B, 1, 1, 1) * mL) + mx.arange(L).reshape(
            1, 1, L, 1
        )
        offsets = offsets * S
        offsets = mx.broadcast_to(offsets, (B, iH, L, 1))
        return mx.take(mask.reshape(B * mL * S), indices + offsets, axis=0)
    if iH == 1 and mH != 1:
        indices = mx.broadcast_to(indices, (B, mH, L, K))
    elif iH != mH:
        raise ValueError("mask head dimension does not match top-k indices")
    offsets = (
        mx.arange(B * mH).reshape(B, mH, 1, 1) * mL
        + mx.arange(L).reshape(1, 1, L, 1)
    ) * S
    return mx.take(mask.reshape(B * mH * mL * S), indices + offsets, axis=0)


def _has_full_topk_causal_prefix(total_context: int, query_length: int, topk: int):
    return total_context - query_length + 1 >= topk


def _causal_topk_prefix_rows(
    total_context: int, query_length: int, topk: int
) -> int:
    """Number of leading query rows whose exact top-k is the causal prefix."""
    query_offset = total_context - query_length
    return min(query_length, max(0, topk - query_offset))


def _index_share_pipeline_boundaries(
    indexer_types: List[str], pipeline_size: int
) -> List[int]:
    """Split layers without starting a pipeline stage on a shared indexer."""
    num_layers = len(indexer_types)
    if pipeline_size < 1:
        raise ValueError("pipeline_size must be positive")
    if pipeline_size > num_layers:
        raise ValueError("pipeline_size cannot exceed the number of layers")
    if not indexer_types or indexer_types[0] != "full":
        raise ValueError("the first GLM DSA layer must own a full indexer")
    if pipeline_size == 1:
        return [0, num_layers]

    candidates = [
        idx
        for idx, indexer_type in enumerate(indexer_types[1:], start=1)
        if indexer_type == "full"
    ]
    if len(candidates) < pipeline_size - 1:
        raise ValueError(
            "pipeline parallelism requires every stage to start on a full "
            "GLM DSA indexer layer"
        )

    boundaries = [0]
    for split in range(1, pipeline_size):
        available = [idx for idx in candidates if idx > boundaries[-1]]
        remaining = pipeline_size - split - 1
        eligible = available[: len(available) - remaining]
        target = round(split * num_layers / pipeline_size)
        boundaries.append(min(eligible, key=lambda idx: (abs(idx - target), idx)))
    boundaries.append(num_layers)
    return boundaries


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
    num_nextn_predict_layers: int = 0
    index_share_for_mtp_iteration: bool = False
    indexer_norm_eps: float = 1e-6
    indexer_rope_interleave: bool = True

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
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError(
                "indexer_types must contain exactly one entry per decoder layer"
            )
        invalid_indexer_types = set(self.indexer_types) - {"full", "shared"}
        if invalid_indexer_types:
            raise ValueError(
                f"unsupported GLM DSA indexer types: {sorted(invalid_indexer_types)}"
            )
        if not self.indexer_types or self.indexer_types[0] != "full":
            raise ValueError("the first GLM DSA layer must own a full indexer")


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        object.__setattr__(self, "_q_a_dense_cache", None)
        indexer_types = config.indexer_types or []
        indexer_type = (
            indexer_types[layer_idx] if layer_idx < len(indexer_types) else "full"
        )
        self.skip_topk = indexer_type == "shared"
        self.share_mtp_iteration_topk = bool(
            config.index_share_for_mtp_iteration
            and layer_idx >= config.num_hidden_layers
        )
        if self.skip_topk:
            self.indexer = None

    def _indexer_head_weights(self, x: mx.array) -> mx.array:
        """Project Indexer head weights in FP32, matching the reference model."""
        indexer = self.indexer
        projection = indexer.weights_proj
        if isinstance(projection, nn.Linear):
            weights = projection(x.astype(projection.weight.dtype)).astype(
                mx.float32
            )
        else:
            # Quantized third-party checkpoints cannot retain an FP32 linear
            # weight. Their projection remains the best available fallback.
            weights = projection(x).astype(mx.float32)
        return weights * (indexer.n_heads**-0.5)

    def _indexer_topk(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
        mask_is_causal: bool = False,
    ):
        indexer = self.indexer
        b, s, _ = x.shape
        q = indexer.wq_b(qr)
        q = q.reshape(b, s, indexer.n_heads, indexer.head_dim).swapaxes(1, 2)
        k = indexer.wk(x)
        k = indexer.k_norm(k)
        k = mx.reshape(k, (b, 1, s, indexer.head_dim))

        offset = cache.offset if cache is not None else 0
        q = indexer.rope(q, offset=offset)
        k = indexer.rope(k, offset=offset)

        if cache is not None:
            k, _ = cache.update_and_fetch(
                k, mx.zeros([b, 1, s, 0], dtype=k.dtype)
            )
        mask = _normalize_attention_mask(mask, q.shape[2], k.shape[2])
        _validate_attention_mask(mask, b, self.num_heads, q.shape[2], k.shape[2])
        if k.shape[2] <= indexer.index_topk:
            return None
        native_indices = self._native_indexer_topk(
            q, x, k, mask, mask_is_causal=mask_is_causal
        )
        if native_indices is not None:
            return native_indices
        if _fast_prefill_enabled() and b == 1 and s > 1:
            # Once the FP32 reference projection disables the 16-bit native
            # Indexer, a dense [B, H, L, S] score tensor is prohibitively large
            # even well below the sparse-attention activation threshold.
            return self._block_indexer_topk(q, x, k, mask)
        return self._dense_indexer_topk(q, x, k, mask)

    def _update_indexer_k_cache(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ):
        indexer = self.indexer
        b, s, _ = x.shape
        k = indexer.wk(x)
        k = indexer.k_norm(k)
        k = mx.reshape(k, (b, 1, s, indexer.head_dim))

        offset = cache.offset if cache is not None else 0
        k = indexer.rope(k, offset=offset)

        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0], dtype=k.dtype))
        return k, offset

    def _dense_indexer_topk(
        self,
        q: mx.array,
        x: mx.array,
        k: mx.array,
        mask: Optional[mx.array],
    ):
        indexer = self.indexer
        mask = _normalize_attention_mask(mask, q.shape[2], k.shape[2])
        _validate_attention_mask(
            mask, q.shape[0], self.num_heads, q.shape[2], k.shape[2]
        )
        scores = (
            q.astype(mx.float32)
            @ k.astype(mx.float32).swapaxes(-1, -2)
        ) * indexer.softmax_scale
        scores = mx.maximum(scores, 0)
        weights = self._indexer_head_weights(x)
        weights = weights.swapaxes(-1, -2)[..., None]
        scores = scores * weights
        scores = scores.sum(axis=1, keepdims=True)
        scores = _apply_attention_mask(scores, mask)
        return mx.argpartition(scores, kth=-indexer.index_topk, axis=-1)[
            ..., -indexer.index_topk :
        ]

    def _native_indexer_decision(
        self,
        q: mx.array,
        x: mx.array,
        k: mx.array,
        mask: Optional[mx.array],
        mask_is_causal: bool = False,
    ):
        indexer = self.indexer
        if not _fast_prefill_enabled():
            return False, "fast_prefill_disabled"
        if not _native_indexer_enabled():
            return False, "disabled"
        if not _native_indexer_available():
            return False, "missing_symbol"
        if isinstance(mask, str):
            if mask != "causal":
                return False, "mask_type"
            mask_is_causal = True
        elif mask is not None:
            if not mask_is_causal:
                return False, "custom_mask"
            if len(mask.shape) not in (2, 4):
                return False, "mask_rank"
        B, H, L, D = q.shape
        if B != 1:
            return False, "batch_size_not_one"
        if L <= 1:
            if not _native_decode_indexer_enabled():
                return False, "decode"
            if not _native_decode_indexer_available():
                return False, "decode_missing_symbol"
        if H != 32:
            return False, f"unsupported_index_heads:{H}"
        if D != 128:
            return False, f"unsupported_index_head_dim:{D}"
        if k.shape[1] != 1:
            return False, "unsupported_kv_heads"
        if k.shape[-1] != D:
            return False, "mixed_index_head_dim"
        if indexer.index_topk != 2048:
            return False, f"unsupported_topk:{indexer.index_topk}"
        if k.shape[2] < 4096:
            return False, "below_native_indexer_min_context"
        if q.dtype not in (mx.float16, mx.bfloat16):
            return False, f"unsupported_dtype:{q.dtype}"
        if k.dtype != q.dtype:
            return False, "mixed_dtype"
        return True, "native_indexer"

    def _native_indexer_topk(
        self,
        q: mx.array,
        x: mx.array,
        k: mx.array,
        mask: Optional[mx.array],
        mask_is_causal: bool = False,
    ):
        use_native, reason = self._native_indexer_decision(
            q, x, k, mask, mask_is_causal=mask_is_causal
        )
        if not use_native:
            _record_native_indexer_decision(False, reason)
            return None

        indexer = self.indexer
        weights = self._indexer_head_weights(x)
        if weights.dtype != mx.float32:
            _record_native_indexer_decision(False, "non_fp32_weight")
            return None
        causal = mask_is_causal or (
            isinstance(mask, str) and mask == "causal"
        )
        query_length = q.shape[2]
        key_length = k.shape[2]
        query_chunk = _native_indexer_query_chunk_size(query_length, key_length)
        if query_chunk == 0:
            _record_native_indexer_decision(False, "score_memory_limit")
            return None
        chunked = query_chunk < query_length
        query_offset = key_length - query_length
        index_chunks = []
        for start in range(0, query_length, query_chunk):
            stop = min(start + query_chunk, query_length)
            q_chunk = q[:, :, start:stop, :]
            weight_chunk = weights[:, start:stop, :]
            use_causal_prefix_shortcut = causal and not chunked
            scores = _profile_stage(
                "native_indexer_scores",
                lambda q_chunk=q_chunk, weight_chunk=weight_chunk, start=start: (
                    _native_indexer_scores(
                        q_chunk,
                        k,
                        weight_chunk,
                        scale=indexer.softmax_scale,
                        causal=causal,
                        skip_causal_future_store=use_causal_prefix_shortcut,
                        causal_q_offset=(query_offset + start) if causal else -1,
                    )
                ),
                inputs=(q_chunk, k, weight_chunk),
            )
            if scores is None:
                _record_native_indexer_decision(False, "scores_unavailable")
                return None
            indices = _profile_stage(
                "native_indexer_topk",
                lambda scores=scores: _native_indexer_topk_indices(
                    scores,
                    indexer.index_topk,
                    bucketed=True,
                    causal_valid_prefix=use_causal_prefix_shortcut,
                ),
                inputs=scores,
            )
            if indices is None:
                _record_native_indexer_decision(False, "topk_unavailable")
                return None
            index_chunks.append(indices)
        indices = (
            index_chunks[0]
            if len(index_chunks) == 1
            else mx.concatenate(index_chunks, axis=2)
        )
        _record_native_indexer_decision(True, reason)
        return indices

    def _block_indexer_topk(
        self,
        q: mx.array,
        x: mx.array,
        k: mx.array,
        mask: Optional[mx.array],
    ):
        indexer = self.indexer
        B, _, L, _ = q.shape
        total_length = k.shape[2]
        mask = _normalize_attention_mask(mask, L, total_length)
        _validate_attention_mask(mask, B, self.num_heads, L, total_length)
        topk = indexer.index_topk
        query_chunk = _fast_prefill_query_chunk_size(L)
        key_block = _fast_prefill_key_block_size(total_length, topk)
        weights = self._indexer_head_weights(x)
        weights = weights.swapaxes(-1, -2)[..., None]

        all_indices = []
        for q_start in range(0, L, query_chunk):
            q_stop = min(q_start + query_chunk, L)
            q_chunk = q[:, :, q_start:q_stop, :]
            q_chunk_fp32 = q_chunk.astype(mx.float32)
            weight_chunk = weights[:, :, q_start:q_stop, :]
            best_scores = None
            best_indices = None

            for k_start in range(0, total_length, key_block):
                k_stop = min(k_start + key_block, total_length)
                k_block = k[:, :, k_start:k_stop, :]
                block_scores = q_chunk_fp32 @ k_block.astype(
                    mx.float32
                ).swapaxes(-1, -2)
                block_scores = block_scores * indexer.softmax_scale
                block_scores = mx.maximum(block_scores, 0)
                block_scores = block_scores * weight_chunk
                block_scores = block_scores.sum(axis=1, keepdims=True)
                mask_block = _slice_attention_mask(
                    mask, q_start, q_stop, k_start, k_stop
                )
                block_scores = _apply_attention_mask(block_scores, mask_block)

                block_len = block_scores.shape[-1]
                if block_len > topk:
                    block_local_indices = mx.argpartition(
                        block_scores, kth=-topk, axis=-1
                    )[..., -topk:]
                    block_scores = mx.take_along_axis(
                        block_scores, block_local_indices, axis=-1
                    )
                else:
                    block_local_indices = mx.broadcast_to(
                        mx.arange(block_len).reshape(1, 1, 1, block_len),
                        block_scores.shape,
                    )
                block_indices = block_local_indices + k_start

                if best_scores is None:
                    best_scores = block_scores
                    best_indices = block_indices
                else:
                    best_scores = mx.concatenate([best_scores, block_scores], axis=-1)
                    best_indices = mx.concatenate(
                        [best_indices, block_indices], axis=-1
                    )

                if best_scores.shape[-1] > topk:
                    keep = mx.argpartition(best_scores, kth=-topk, axis=-1)[
                        ..., -topk:
                    ]
                    best_scores = mx.take_along_axis(best_scores, keep, axis=-1)
                    best_indices = mx.take_along_axis(best_indices, keep, axis=-1)

            all_indices.append(best_indices)

        return mx.concatenate(all_indices, axis=2)

    def _fast_prefill_decision(
        self,
        *,
        B: int,
        L: int,
        cache: Optional[Any],
        topk_indices: Optional[mx.array],
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
        supported_cache_types = (
            GlmMlaKVCache,
            QuantizedGlmMlaKVCache,
            BatchGlmMlaKVCache,
            BatchQuantizedGlmMlaKVCache,
        )
        if not isinstance(kv_cache, supported_cache_types):
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
        if K > k_pe.shape[2]:
            return False, "topk_exceeds_context"
        short_quantized_verify = (
            isinstance(kv_cache, QuantizedGlmMlaKVCache)
            and L <= _MAX_QUANTIZED_SPARSE_VERIFY_TOKENS
        )
        if (
            not short_quantized_verify
            and not _sparse_prefill_context_ready(k_pe.shape[2])
        ):
            return False, "below_sparse_min_context"
        if not _has_full_topk_causal_prefix(k_pe.shape[2], L, K):
            return False, "causal_prefix_shorter_than_topk"
        if k_pe.shape[1] != 1:
            return False, "unsupported_kv_heads"
        offset = _scalar_int(kv_cache.offset)
        if offset is None:
            return False, "non_scalar_offset"
        _warn_fast_prefill_large_topk(K)
        return True, (
            "short_quantized_verify" if short_quantized_verify else "fast"
        )

    def _native_sparse_prefill_decision(
        self,
        *,
        B: int,
        L: int,
        kv_cache: Any,
        kv_latent: Any,
        k_pe: mx.array,
        topk_indices: mx.array,
        mask: Optional[mx.array] = None,
        mask_is_causal: bool = False,
    ):
        if not _fast_prefill_enabled():
            return False, "fast_prefill_disabled"
        if not _native_sparse_prefill_enabled():
            return False, "disabled"
        if _native_sparse_mla_kernel() is None:
            return False, "missing_symbol"
        if isinstance(mask, str):
            if mask != "causal":
                return False, "mask_type"
            mask_is_causal = True
        if not mask_is_causal:
            return False, "custom_mask"
        if L <= 1:
            return False, "decode"
        if len(topk_indices.shape) != 4:
            return False, "topk_rank"
        if topk_indices.shape[0] != B or topk_indices.shape[2] != L:
            return False, "topk_shape"
        quantized_kv = isinstance(kv_cache, QuantizedGlmMlaKVCache)
        if not isinstance(kv_cache, GlmMlaKVCache):
            if quantized_kv:
                if not _native_sparse_prefill_quantized_kv_enabled():
                    return False, "quantized_kv"
            elif isinstance(kv_cache, BatchQuantizedGlmMlaKVCache):
                return False, "batched_quantized_kv_cache"
            elif isinstance(kv_cache, BatchGlmMlaKVCache):
                return False, "batched_kv_cache"
            else:
                return False, f"unsupported_cache:{type(kv_cache).__name__}"
        if isinstance(kv_latent, (tuple, list)) and not quantized_kv:
            return False, "quantized_kv_state"
        if B != 1:
            return False, "batch_size_not_one"
        if topk_indices.shape[1] != 1:
            return False, "topk_heads"
        if self.num_heads != 64:
            return False, "unsupported_heads"
        rope_dim = k_pe.shape[-1]
        if rope_dim != 64:
            return False, f"unsupported_rope_dim:{rope_dim}"
        if k_pe.shape[1] != 1:
            return False, "unsupported_kv_heads"
        if quantized_kv:
            if kv_cache.bits != 8:
                return False, f"unsupported_kv_bits:{kv_cache.bits}"
            if kv_cache.group_size != 64:
                return False, f"unsupported_kv_group_size:{kv_cache.group_size}"
            max_context = _native_sparse_prefill_quantized_kv_max_context_length()
            if max_context and k_pe.shape[2] > max_context:
                return False, "quantized_kv_context_exceeds_limit"
        else:
            if kv_latent.shape[1] != 1:
                return False, "unsupported_kv_heads"
            if kv_latent.shape[-1] != 512:
                return False, f"unsupported_latent_dim:{kv_latent.shape[-1]}"
            if kv_latent.dtype not in (mx.float16, mx.bfloat16):
                return False, f"unsupported_kv_dtype:{kv_latent.dtype}"
            if k_pe.dtype != kv_latent.dtype:
                return False, "mixed_kv_dtype"
        topk = topk_indices.shape[-1]
        if topk != 2048:
            return False, f"unsupported_topk:{topk}"
        if topk > k_pe.shape[2]:
            return False, "topk_exceeds_context"
        if k_pe.shape[2] < _native_sparse_prefill_min_context_length():
            return False, "below_native_sparse_min_context"
        return (
            True,
            "native_sparse_mla_quantized_kv" if quantized_kv else "native_sparse_mla",
        )

    def _native_q8_vup_decision(self, x: mx.array):
        if not _native_q8_vup_enabled():
            return False, "disabled"
        if _native_q8_vup_kernel() is None:
            return False, "missing_symbol"
        if not isinstance(self.unembed_out, QuantizedMultiLinear):
            return False, "unquantized_unembed_out"
        if self.unembed_out.bits != 8:
            return False, f"unsupported_bits:{self.unembed_out.bits}"
        if self.unembed_out.group_size != 64:
            return False, f"unsupported_group_size:{self.unembed_out.group_size}"
        if self.unembed_out.mode != "affine":
            return False, f"unsupported_mode:{self.unembed_out.mode}"
        weight = self.unembed_out["weight"]
        scales = self.unembed_out["scales"]
        biases = self.unembed_out.get("biases")
        if biases is None:
            return False, "missing_biases"
        if len(x.shape) != 4:
            return False, "input_rank"
        B, H, L, K = x.shape
        if H != 64:
            return False, f"unsupported_heads:{H}"
        if K != 512:
            return False, f"unsupported_latent_dim:{K}"
        if weight.dtype != mx.uint32:
            return False, f"unsupported_weight_dtype:{weight.dtype}"
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False, "mixed_dtype"
        if len(weight.shape) != 3 or len(scales.shape) != 3 or len(biases.shape) != 3:
            return False, "weight_rank"
        if weight.shape[0] != H or scales.shape[0] != H or biases.shape[0] != H:
            return False, "weight_heads"
        if scales.shape[1] != 256 or biases.shape[1] != 256:
            return False, "unsupported_value_dim"
        if weight.shape[2] * 4 != K:
            return False, "weight_latent_dim"
        if scales.shape[2] != K // 64 or biases.shape[2] != K // 64:
            return False, "weight_group_shape"
        return True, "native_q8_vup"

    def _native_q4_vup_decision(self, x: mx.array):
        if not _native_q4_vup_enabled():
            return False, "disabled"
        if _native_q4_vup_kernel() is None:
            return False, "missing_symbol"
        if not isinstance(self.unembed_out, QuantizedMultiLinear):
            return False, "unquantized_unembed_out"
        if self.unembed_out.bits != 4:
            return False, f"unsupported_bits:{self.unembed_out.bits}"
        if self.unembed_out.group_size != 64:
            return False, f"unsupported_group_size:{self.unembed_out.group_size}"
        if self.unembed_out.mode != "affine":
            return False, f"unsupported_mode:{self.unembed_out.mode}"
        weight = self.unembed_out["weight"]
        scales = self.unembed_out["scales"]
        biases = self.unembed_out.get("biases")
        if biases is None:
            return False, "missing_biases"
        if len(x.shape) != 4:
            return False, "input_rank"
        B, H, L, K = x.shape
        if H != 64:
            return False, f"unsupported_heads:{H}"
        if K != 512:
            return False, f"unsupported_latent_dim:{K}"
        if weight.dtype != mx.uint32:
            return False, f"unsupported_weight_dtype:{weight.dtype}"
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False, "mixed_dtype"
        if len(weight.shape) != 3 or len(scales.shape) != 3 or len(biases.shape) != 3:
            return False, "weight_rank"
        if weight.shape[0] != H or scales.shape[0] != H or biases.shape[0] != H:
            return False, "weight_heads"
        if scales.shape[1] != 256 or biases.shape[1] != 256:
            return False, "unsupported_value_dim"
        if weight.shape[2] * 8 != K:
            return False, "weight_latent_dim"
        if scales.shape[2] != K // 64 or biases.shape[2] != K // 64:
            return False, "weight_group_shape"
        return True, "native_q4_vup"

    def _q_a_dense_cache_decision(self, x: mx.array):
        if not _q_a_dense_cache_enabled():
            return False, "disabled"
        if not hasattr(self.q_a_proj, "bits"):
            return False, "unquantized_q_a_proj"
        if self.q_a_proj.bits != 4:
            return False, f"unsupported_bits:{self.q_a_proj.bits}"
        if self.q_a_proj.group_size != 64:
            return False, f"unsupported_group_size:{self.q_a_proj.group_size}"
        if self.q_a_proj.mode != "affine":
            return False, f"unsupported_mode:{self.q_a_proj.mode}"
        weight = self.q_a_proj["weight"]
        scales = self.q_a_proj["scales"]
        biases = self.q_a_proj.get("biases")
        if biases is None:
            return False, "missing_biases"
        if len(x.shape) != 3:
            return False, "input_rank"
        if x.shape[-1] != 6144:
            return False, f"unsupported_input_dim:{x.shape[-1]}"
        if self.q_lora_rank != 2048:
            return False, f"unsupported_q_lora_rank:{self.q_lora_rank}"
        if weight.dtype != mx.uint32:
            return False, f"unsupported_weight_dtype:{weight.dtype}"
        if x.dtype not in (mx.float16, mx.bfloat16):
            return False, f"unsupported_dtype:{x.dtype}"
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False, "mixed_dtype"
        if len(weight.shape) != 2 or len(scales.shape) != 2 or len(biases.shape) != 2:
            return False, "weight_rank"
        if weight.shape[0] != self.q_lora_rank:
            return False, "weight_output_dim"
        if scales.shape[0] != weight.shape[0] or biases.shape[0] != weight.shape[0]:
            return False, "scale_output_dim"
        if weight.shape[1] * 8 != x.shape[-1]:
            return False, "weight_input_dim"
        if scales.shape[1] != x.shape[-1] // 64:
            return False, "scale_group_shape"
        if biases.shape[1] != x.shape[-1] // 64:
            return False, "bias_group_shape"
        return True, "q_a_dense_cache"

    def _q_a_dense_cache_key(self, x: mx.array):
        return (
            x.dtype,
            id(self.q_a_proj["weight"]),
            id(self.q_a_proj["scales"]),
            id(self.q_a_proj.get("biases")),
        )

    def _q_a_dense_weight(self, x: mx.array):
        key = self._q_a_dense_cache_key(x)
        cache = object.__getattribute__(self, "_q_a_dense_cache")
        if cache is not None and cache[0] == key:
            return cache[1], False

        def dequantize_weight():
            dense_weight = mx.dequantize(
                self.q_a_proj["weight"],
                scales=self.q_a_proj["scales"],
                biases=self.q_a_proj["biases"],
                group_size=self.q_a_proj.group_size,
                bits=self.q_a_proj.bits,
                mode=self.q_a_proj.mode,
            )
            if not _prefill_profile_enabled():
                mx.eval(dense_weight)
            return dense_weight

        dense_weight = _profile_stage(
            "q_a_dense_cache_dequantization",
            dequantize_weight,
        )
        object.__setattr__(self, "_q_a_dense_cache", (key, dense_weight))
        return dense_weight, True

    def _q_a_dense_cache_project(self, x: mx.array):
        use_dense, reason = self._q_a_dense_cache_decision(x)
        if not use_dense:
            if hasattr(self.q_a_proj, "bits"):
                _record_q_a_dense_cache_decision(False, reason)
            return None
        try:
            dense_weight, built = self._q_a_dense_weight(x)
            output = _profile_stage(
                "q_a_dense_projection",
                lambda: x @ dense_weight.T,
            )
            _record_q_a_dense_cache_decision(True, reason, built=built)
            return output
        except Exception as exc:
            _record_q_a_dense_cache_decision(
                False,
                f"runtime_error:{type(exc).__name__}",
            )
            return None

    def _native_q4_qa_decision(self, x: mx.array):
        if not _native_q4_qa_enabled():
            return False, "disabled"
        if _native_q4_qa_kernel() is None:
            return False, "missing_symbol"
        if not hasattr(self.q_a_proj, "bits"):
            return False, "unquantized_q_a_proj"
        if self.q_a_proj.bits != 4:
            return False, f"unsupported_bits:{self.q_a_proj.bits}"
        if self.q_a_proj.group_size != 64:
            return False, f"unsupported_group_size:{self.q_a_proj.group_size}"
        if self.q_a_proj.mode != "affine":
            return False, f"unsupported_mode:{self.q_a_proj.mode}"
        weight = self.q_a_proj["weight"]
        scales = self.q_a_proj["scales"]
        biases = self.q_a_proj.get("biases")
        if biases is None:
            return False, "missing_biases"
        if len(x.shape) != 3:
            return False, "input_rank"
        if x.shape[-1] != 6144:
            return False, f"unsupported_input_dim:{x.shape[-1]}"
        if self.q_lora_rank != 2048:
            return False, f"unsupported_q_lora_rank:{self.q_lora_rank}"
        if weight.dtype != mx.uint32:
            return False, f"unsupported_weight_dtype:{weight.dtype}"
        if x.dtype not in (mx.float16, mx.bfloat16):
            return False, f"unsupported_dtype:{x.dtype}"
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False, "mixed_dtype"
        if len(weight.shape) != 2 or len(scales.shape) != 2 or len(biases.shape) != 2:
            return False, "weight_rank"
        if weight.shape[0] != self.q_lora_rank:
            return False, "weight_output_dim"
        if scales.shape[0] != weight.shape[0] or biases.shape[0] != weight.shape[0]:
            return False, "scale_output_dim"
        if weight.shape[1] * 8 != x.shape[-1]:
            return False, "weight_input_dim"
        if scales.shape[1] != x.shape[-1] // 64:
            return False, "scale_group_shape"
        if biases.shape[1] != x.shape[-1] // 64:
            return False, "bias_group_shape"
        return True, "native_q4_qa"

    def _q_a_project(self, x: mx.array):
        dense_output = self._q_a_dense_cache_project(x)
        if dense_output is not None:
            return dense_output
        use_native, reason = self._native_q4_qa_decision(x)
        if use_native:
            try:
                kernel = _native_q4_qa_kernel()
                output = _profile_stage(
                    "native_q4_qa_projection",
                    lambda: kernel(
                        x,
                        self.q_a_proj["weight"],
                        self.q_a_proj["scales"],
                        self.q_a_proj["biases"],
                    ),
                )
                _record_native_q4_qa_decision(True, reason)
                return output
            except Exception as exc:
                reason = f"runtime_error:{type(exc).__name__}"

        if hasattr(self.q_a_proj, "bits"):
            _record_native_q4_qa_decision(False, reason)
        return self.q_a_proj(x)

    def _native_q4_qb_decision(self, x: mx.array):
        if not _native_q4_qb_enabled():
            return False, "disabled"
        if _native_q4_qb_kernel() is None:
            return False, "missing_symbol"
        if not hasattr(self.q_b_proj, "bits"):
            return False, "unquantized_q_b_proj"
        if self.q_b_proj.bits != 4:
            return False, f"unsupported_bits:{self.q_b_proj.bits}"
        if self.q_b_proj.group_size != 64:
            return False, f"unsupported_group_size:{self.q_b_proj.group_size}"
        if self.q_b_proj.mode != "affine":
            return False, f"unsupported_mode:{self.q_b_proj.mode}"
        weight = self.q_b_proj["weight"]
        scales = self.q_b_proj["scales"]
        biases = self.q_b_proj.get("biases")
        if biases is None:
            return False, "missing_biases"
        if len(x.shape) != 3:
            return False, "input_rank"
        if x.shape[-1] != 2048:
            return False, f"unsupported_input_dim:{x.shape[-1]}"
        if self.num_heads != 64:
            return False, f"unsupported_heads:{self.num_heads}"
        if self.q_head_dim != 256:
            return False, f"unsupported_q_head_dim:{self.q_head_dim}"
        if weight.dtype != mx.uint32:
            return False, f"unsupported_weight_dtype:{weight.dtype}"
        if x.dtype not in (mx.float16, mx.bfloat16):
            return False, f"unsupported_dtype:{x.dtype}"
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False, "mixed_dtype"
        if len(weight.shape) != 2 or len(scales.shape) != 2 or len(biases.shape) != 2:
            return False, "weight_rank"
        if weight.shape[0] != self.num_heads * self.q_head_dim:
            return False, "weight_output_dim"
        if scales.shape[0] != weight.shape[0] or biases.shape[0] != weight.shape[0]:
            return False, "scale_output_dim"
        if weight.shape[1] * 8 != x.shape[-1]:
            return False, "weight_input_dim"
        if scales.shape[1] != x.shape[-1] // 64:
            return False, "scale_group_shape"
        if biases.shape[1] != x.shape[-1] // 64:
            return False, "bias_group_shape"
        return True, "native_q4_qb"

    def _q_b_project(self, x: mx.array):
        use_native, reason = self._native_q4_qb_decision(x)
        if use_native:
            try:
                kernel = _native_q4_qb_kernel()
                output = _profile_stage(
                    "native_q4_qb_projection",
                    lambda: kernel(
                        x,
                        self.q_b_proj["weight"],
                        self.q_b_proj["scales"],
                        self.q_b_proj["biases"],
                    ),
                    inputs=x,
                )
                _record_native_q4_qb_decision(True, reason)
                return output
            except Exception as exc:
                reason = f"runtime_error:{type(exc).__name__}"

        if hasattr(self.q_b_proj, "bits"):
            _record_native_q4_qb_decision(False, reason)
        return self.q_b_proj(x)

    def _unembed_out_project(self, x: mx.array):
        use_native, reason = self._native_q4_vup_decision(x)
        if use_native:
            try:
                kernel = _native_q4_vup_kernel()
                flat = _profile_stage(
                    "native_q4_vup",
                    lambda: kernel(
                        x,
                        self.unembed_out["weight"],
                        self.unembed_out["scales"],
                        self.unembed_out["biases"],
                    ),
                    inputs=x,
                )
                B, H, L, _ = x.shape
                V = flat.shape[-1] // H
                output = flat.reshape(B, L, H, V).transpose(0, 2, 1, 3)
                _record_native_q4_vup_decision(True, reason)
                return output
            except Exception as exc:
                reason = f"runtime_error:{type(exc).__name__}"

        if isinstance(self.unembed_out, QuantizedMultiLinear):
            _record_native_q4_vup_decision(False, reason)

        use_native, reason = self._native_q8_vup_decision(x)
        if use_native:
            try:
                kernel = _native_q8_vup_kernel()
                flat = _profile_stage(
                    "native_q8_vup",
                    lambda: kernel(
                        x,
                        self.unembed_out["weight"],
                        self.unembed_out["scales"],
                        self.unembed_out["biases"],
                    ),
                    inputs=x,
                )
                B, H, L, _ = x.shape
                V = flat.shape[-1] // H
                output = flat.reshape(B, L, H, V).transpose(0, 2, 1, 3)
                _record_native_q8_vup_decision(True, reason)
                return output
            except Exception as exc:
                reason = f"runtime_error:{type(exc).__name__}"

        if isinstance(self.unembed_out, QuantizedMultiLinear):
            _record_native_q8_vup_decision(False, reason)
        return _profile_stage(
            "latent_kv_projection",
            lambda: self.unembed_out(x),
            inputs=x,
        )

    def _dense_sparse_mask(self, mask, topk_indices, key_length: int):
        shape = list(topk_indices.shape)
        shape[-1] = key_length
        sparse_mask = mx.zeros(shape, dtype=mx.bool_)
        sparse_mask = mx.put_along_axis(
            sparse_mask, topk_indices, mx.array(True), axis=-1
        )
        if mask is None:
            return sparse_mask
        if mask.dtype == mx.bool_:
            return sparse_mask & mask
        if not mx.issubdtype(mask.dtype, mx.floating):
            raise ValueError("custom attention masks must be boolean or floating point")
        return mx.where(
            sparse_mask,
            mask,
            mx.array(-float("inf"), dtype=mask.dtype),
        )

    def _native_sparse_prefill_attention(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_cache: Any,
        kv_latent: Any,
        k_pe: mx.array,
        topk_indices: mx.array,
    ):
        kernel = _native_sparse_mla_kernel()
        if kernel is None:
            return None, "missing_symbol"
        topk = (
            topk_indices
            if topk_indices.dtype == mx.uint32
            else topk_indices.astype(mx.uint32)
        )
        try:
            B, _H, L, K = topk.shape
            total_context = k_pe.shape[2]
            query_offset = total_context - L
            prefix_rows = _causal_topk_prefix_rows(total_context, L, K)
            topk_length = None
            kernel_kwargs = {"causal": True}
            q_latent = _profile_stage(
                "latent_kv_projection",
                lambda: self.embed_q(q_nope),
                inputs=q_nope,
            )
            if isinstance(kv_cache, QuantizedGlmMlaKVCache):
                # Verification batches are tiny but may have a very long cache.
                # Compact their already-causal top-k rows before dequantizing so
                # native attention never materializes the full float KV cache.
                # An early causal row has fewer than K valid keys, so replace
                # padded indexer output with its exact 0..q_abs prefix.  The
                # kernel length vector prevents the unused compact slots from
                # receiving attention mass.
                if prefix_rows:
                    causal_prefix = mx.arange(K, dtype=mx.uint32).reshape(
                        1, 1, 1, K
                    )
                    causal_prefix = mx.broadcast_to(
                        causal_prefix, (B, 1, prefix_rows, K)
                    )
                    topk = mx.concatenate(
                        [causal_prefix, topk[:, :, prefix_rows:, :]], axis=2
                    )
                    valid_lengths = mx.minimum(
                        mx.arange(L, dtype=mx.uint32)
                        + mx.array(query_offset + 1, dtype=mx.uint32),
                        mx.array(K, dtype=mx.uint32),
                    )
                    topk_length = mx.broadcast_to(
                        valid_lengths.reshape(1, L), (B, L)
                    )
                selected_latent = self._gather_cached_latent(
                    kv_cache, kv_latent, topk
                )
                selected_k_pe = _gather_sequence_by_flat_index(k_pe, topk)
                B, _H, L, K, D = selected_latent.shape
                kv_latent = selected_latent.reshape(B, 1, L * K, D)
                k_pe = selected_k_pe.reshape(B, 1, L * K, k_pe.shape[-1])
                topk = mx.arange(L * K, dtype=mx.uint32).reshape(1, 1, L, K)
                if B != 1:
                    topk = mx.broadcast_to(topk, (B, 1, L, K))
                kernel_kwargs["causal"] = False
                if topk_length is not None:
                    kernel_kwargs["topk_length"] = topk_length
                quantized_reason = "native_sparse_mla_quantized_kv_compact"
            else:
                if prefix_rows:
                    # The native kernel can synthesize exact early causal rows.
                    # Drop their padded indexer rows, then resume the real top-k
                    # once each query has at least K causal keys.
                    if prefix_rows < L:
                        topk = topk[:, :, prefix_rows:, :]
                        kernel_kwargs["causal_prefix_rows"] = prefix_rows
                    kernel_kwargs["topk_valid_prefix"] = True
                    kernel_kwargs["causal_prefix_indices"] = True
                quantized_reason = "native_sparse_mla"
            profile_inputs = (q_latent, q_pe, kv_latent, k_pe, topk)
            if topk_length is not None:
                profile_inputs = (*profile_inputs, topk_length)
            output = _profile_stage(
                "native_sparse_attention",
                lambda: kernel(
                    q_latent,
                    q_pe,
                    kv_latent,
                    k_pe,
                    topk,
                    self.scale,
                    **kernel_kwargs,
                ),
                inputs=profile_inputs,
            )
            return (
                self._unembed_out_project(output),
                quantized_reason,
            )
        except Exception as exc:
            return None, f"runtime_error:{type(exc).__name__}"

    def _fast_sparse_prefill_attention(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_cache: Any,
        kv_latent: Any,
        k_pe: mx.array,
        topk_indices: mx.array,
        mask: Optional[mx.array],
        mask_is_causal: bool = False,
    ):
        _, _, L, _ = q_nope.shape
        query_chunk = _fast_prefill_query_chunk_size(L)
        topk = topk_indices.shape[-1]
        gather_mask = mask
        if (
            mask_is_causal
            and isinstance(kv_cache, (GlmMlaKVCache, QuantizedGlmMlaKVCache))
            and _has_full_topk_causal_prefix(k_pe.shape[2], L, topk)
        ):
            gather_mask = None

        outputs = []
        for start in range(0, L, query_chunk):
            stop = min(start + query_chunk, L)
            chunk_topk = topk_indices[:, :, start:stop, :]

            latent_selected, k_pe_selected, mask_selected = _profile_stage(
                "sparse_gather",
                lambda chunk_topk=chunk_topk: (
                    self._gather_cached_latent(kv_cache, kv_latent, chunk_topk),
                    _gather_sequence_by_flat_index(k_pe, chunk_topk),
                    _gather_attention_mask(
                        None if gather_mask is None else _slice_attention_mask(
                            gather_mask, start, stop, 0, k_pe.shape[2]
                        ),
                        chunk_topk,
                    ),
                ),
                inputs=(kv_latent, k_pe, chunk_topk, gather_mask),
            )

            q_nope_chunk = q_nope[:, :, start:stop, :]
            q_pe_chunk = q_pe[:, :, start:stop, :]
            outputs.append(
                _profile_stage(
                    "attention",
                    lambda q_nope_chunk=q_nope_chunk,
                    q_pe_chunk=q_pe_chunk,
                    latent_selected=latent_selected,
                    k_pe_selected=k_pe_selected: self._fast_sparse_attention_chunk(
                        q_nope_chunk,
                        q_pe_chunk,
                        latent_selected,
                        k_pe_selected,
                        mask_selected,
                    ),
                    inputs=(
                        q_nope_chunk,
                        q_pe_chunk,
                        latent_selected,
                        k_pe_selected,
                        mask_selected,
                    ),
                )
            )

        return mx.concatenate(outputs, axis=2)

    def _gather_cached_latent(
        self,
        kv_cache: Any,
        kv_latent: Any,
        topk_indices: mx.array,
    ):
        if isinstance(kv_cache, (QuantizedGlmMlaKVCache, BatchQuantizedGlmMlaKVCache)):
            selected = tuple(
                _gather_sequence_by_flat_index(q, topk_indices) for q in kv_latent
            )
            return _profile_stage(
                "latent_kv_dequantization",
                lambda selected=selected: kv_cache.dequantize_keys(selected),
                inputs=selected,
            )
        return _gather_sequence_by_flat_index(kv_latent, topk_indices)

    def _fast_sparse_attention_chunk(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        latent_selected: mx.array,
        k_pe_selected: mx.array,
        mask_selected: Optional[mx.array],
    ):
        B, H, L, D = q_nope.shape
        K = latent_selected.shape[-2]
        R = latent_selected.shape[-1]
        if latent_selected.shape[1] == 1 and H != 1:
            latent_selected = mx.broadcast_to(latent_selected, (B, H, L, K, R))
        pe_scores = mx.sum(
            (q_pe * self.scale)[..., None, :] * k_pe_selected,
            axis=-1,
        )
        pe_scores = _apply_attention_mask(pe_scores, mask_selected)
        q_nope = _profile_stage(
            "latent_kv_projection",
            lambda: self.embed_q(q_nope),
        )
        output = mx.fast.scaled_dot_product_attention(
            q_nope.reshape(B * H * L, 1, 1, R),
            latent_selected.reshape(B * H * L, 1, K, R),
            latent_selected.reshape(B * H * L, 1, K, R),
            scale=self.scale,
            mask=pe_scores.reshape(B * H * L, 1, 1, K),
        )
        output = output.reshape(B, H, L, R)
        return self._unembed_out_project(output)

    def _short_verify_attention(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_cache: Any,
        kv_latent: Any,
        k_pe: mx.array,
        topk_indices: mx.array,
        mask: Optional[mx.array],
    ):
        latent_selected = self._gather_cached_latent(
            kv_cache, kv_latent, topk_indices
        )
        k_pe_selected = _gather_sequence_by_flat_index(k_pe, topk_indices)
        mask_selected = _gather_attention_mask(mask, topk_indices)
        outputs = []
        for index in range(q_nope.shape[2]):
            q_nope_step = self.embed_q(q_nope[:, :, index : index + 1, :])
            q_pe_step = q_pe[:, :, index : index + 1, :]
            latent_step = latent_selected[:, :, index, :, :]
            k_pe_step = k_pe_selected[:, :, index, :, :]
            pe_scores = (q_pe_step * self.scale) @ k_pe_step.swapaxes(-1, -2)
            selected_step_mask = (
                None
                if mask_selected is None
                else mask_selected[:, :, index : index + 1, :]
            )
            pe_scores = _apply_attention_mask(pe_scores, selected_step_mask)
            output = scaled_dot_product_attention(
                q_nope_step,
                latent_step,
                latent_step,
                cache=None,
                scale=self.scale,
                mask=pe_scores,
            )
            outputs.append(self._unembed_out_project(output))
        return mx.concatenate(outputs, axis=2)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
        mask_is_causal: bool = False,
    ):
        B, L, D = x.shape
        profile_total = _prefill_profile_enabled() and L > 1
        profile_decode_total = _GLM_DSA_PROFILE_SCOPE.get() == "decode"
        total_start = (
            time.perf_counter() if profile_total or profile_decode_total else None
        )

        def project_q():
            q_a = _profile_stage(
                "q_a_projection",
                lambda: self._q_a_project(x),
                inputs=x,
            )
            qr = _profile_stage(
                "q_a_layernorm",
                lambda: self.q_a_layernorm(q_a),
                inputs=q_a,
            )
            q = _profile_stage(
                "q_b_projection",
                lambda: self._q_b_project(qr),
                inputs=qr,
            )
            q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(
                0, 2, 1, 3
            )
            q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
            return qr, q_nope, q_pe

        qr, q_nope, q_pe = _profile_stage("q_projection", project_q, inputs=x)

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

        (kv_latent, k_pe), q_pe = _profile_stage(
            "kv_cache_update",
            update_kv_cache,
            inputs=(x, q_pe),
        )
        mask = _normalize_attention_mask(mask, L, k_pe.shape[2])
        _validate_attention_mask(mask, B, self.num_heads, L, k_pe.shape[2])
        mask_valid_rows = None
        if not mask_is_causal:
            mask_valid_rows = _attention_mask_valid_rows(mask)
            mask = _ensure_attention_mask_has_valid_key(mask, mask_valid_rows)

        kv_cache = cache[0] if cache is not None else None
        kv_latent_dequantized = not hasattr(kv_cache, "dequantize_keys")

        def ensure_kv_latent_dequantized():
            nonlocal kv_latent, kv_latent_dequantized
            if kv_latent_dequantized:
                return kv_latent
            kv_latent = _profile_stage(
                "latent_kv_dequantization",
                lambda: kv_cache.dequantize_keys(kv_latent),
                inputs=kv_latent,
            )
            kv_latent_dequantized = True
            return kv_latent

        if cache is None:
            cache = [None] * 2

        if (
            self.share_mtp_iteration_topk
            and prev_topk_indices is not None
            and mask_is_causal
        ):
            if cache[1] is not None:
                _profile_stage(
                    "dsa_indexer_cache_update",
                    lambda: self._update_indexer_k_cache(x, cache[1]),
                    inputs=x,
                )
            topk_indices = prev_topk_indices
        elif self.indexer is not None:
            topk_indices = _profile_stage(
                "dsa_indexer_topk",
                lambda: self._indexer_topk(
                    x,
                    qr,
                    mask,
                    cache=cache[1],
                    mask_is_causal=mask_is_causal,
                ),
                inputs=(x, qr, mask),
            )
        else:
            if prev_topk_indices is None and k_pe.shape[2] > self.config.index_topk:
                raise ValueError(
                    "shared GLM DSA layer requires top-k indices from the "
                    "preceding full indexer layer"
                )
            topk_indices = prev_topk_indices

        fast_sparse_prefill = False
        native_sparse_prefill = False
        short_verify = False
        native_sparse_prefill_reason = None
        dense_sparse_mask_applied = False
        if topk_indices is not None:
            if L == 1:
                _record_fast_prefill_decision(False, "decode")
                gathered_latent = self._gather_cached_latent(
                    kv_cache, kv_latent, topk_indices
                )
                kv_latent = gathered_latent[:, :, 0, :, :]
                kv_latent_dequantized = True
                k_pe = _gather_sequence_by_flat_index(k_pe, topk_indices)[
                    :, :, 0, :, :
                ]
                if mask is not None:
                    mask = _gather_attention_mask(mask, topk_indices)
            else:
                full_topk_causal_prefix = _has_full_topk_causal_prefix(
                    k_pe.shape[2], L, topk_indices.shape[-1]
                )
                short_verify = (
                    _fast_prefill_enabled()
                    and _GLM_DSA_MTP_VERIFY_SCOPE.get()
                    and isinstance(
                        kv_cache, (GlmMlaKVCache, QuantizedGlmMlaKVCache)
                    )
                    and L <= _MAX_QUANTIZED_SPARSE_VERIFY_TOKENS
                    and full_topk_causal_prefix
                )
                if short_verify:
                    _record_native_sparse_prefill_decision(
                        False, "short_verify_split_attention"
                    )
                    _record_fast_prefill_decision(
                        True, "short_verify_split_attention"
                    )
                else:
                    native_sparse_prefill, native_sparse_prefill_reason = (
                        self._native_sparse_prefill_decision(
                            B=B,
                            L=L,
                            kv_cache=kv_cache,
                            kv_latent=kv_latent,
                            k_pe=k_pe,
                            topk_indices=topk_indices,
                            mask=mask,
                            mask_is_causal=mask_is_causal,
                        )
                    )
                if not short_verify and not native_sparse_prefill:
                    _record_native_sparse_prefill_decision(
                        False, native_sparse_prefill_reason
                    )
                    fast_sparse_prefill, reason = self._fast_prefill_decision(
                        B=B,
                        L=L,
                        cache=cache,
                        topk_indices=topk_indices,
                        k_pe=k_pe,
                    )
                    _record_fast_prefill_decision(fast_sparse_prefill, reason)
                if (
                    not short_verify
                    and not native_sparse_prefill
                    and not fast_sparse_prefill
                ):
                    ensure_kv_latent_dequantized()
                    mask = self._dense_sparse_mask(mask, topk_indices, k_pe.shape[2])
                    dense_sparse_mask_applied = True
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

        output = None
        if short_verify:
            output = _profile_stage(
                "attention",
                lambda: self._short_verify_attention(
                    q_nope,
                    q_pe,
                    kv_cache,
                    kv_latent,
                    k_pe,
                    topk_indices,
                    mask,
                ),
                inputs=(q_nope, q_pe, kv_latent, k_pe, topk_indices, mask),
            )
        elif native_sparse_prefill:
            output, native_sparse_prefill_reason = (
                self._native_sparse_prefill_attention(
                    q_nope,
                    q_pe,
                    kv_cache,
                    kv_latent,
                    k_pe,
                    topk_indices,
                )
            )
            if output is not None:
                _record_native_sparse_prefill_decision(
                    True, native_sparse_prefill_reason
                )
            else:
                _record_native_sparse_prefill_decision(
                    False, native_sparse_prefill_reason
                )
                fast_sparse_prefill, reason = self._fast_prefill_decision(
                    B=B,
                    L=L,
                    cache=cache,
                    topk_indices=topk_indices,
                    k_pe=k_pe,
                )
                _record_fast_prefill_decision(fast_sparse_prefill, reason)
                if not fast_sparse_prefill and not dense_sparse_mask_applied:
                    ensure_kv_latent_dequantized()
                    mask = self._dense_sparse_mask(mask, topk_indices, k_pe.shape[2])
                    dense_sparse_mask_applied = True

        if output is None and fast_sparse_prefill:
            output = self._fast_sparse_prefill_attention(
                q_nope,
                q_pe,
                kv_cache,
                kv_latent,
                k_pe,
                topk_indices,
                mask,
                mask_is_causal=mask_is_causal,
            )
        if output is None:
            ensure_kv_latent_dequantized()
            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            pe_scores = _apply_attention_mask(pe_scores, mask)

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
                    inputs=kv_latent,
                )

            output = _profile_stage(
                "attention",
                lambda: scaled_dot_product_attention(
                    q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
                ),
                inputs=(q_nope, k, v, pe_scores),
            )
            if L == 1:
                output = self._unembed_out_project(output)

        if mask_valid_rows is not None:
            output = mx.where(mask_valid_rows, output, mx.zeros_like(output))
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = _profile_stage(
            "o_projection", lambda: self.o_proj(output), inputs=output
        )
        if profile_decode_total:
            _eval_profile_value(output)
            _record_decode_stage(
                "total_decode_attention", time.perf_counter() - total_start
            )
        elif profile_total:
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
        mask_is_causal: bool = False,
    ):
        profile_decode_total = _GLM_DSA_PROFILE_SCOPE.get() == "decode"
        total_start = time.perf_counter() if profile_decode_total else None
        normed_x = _profile_stage(
            "input_layernorm", lambda: self.input_layernorm(x), inputs=x
        )
        r, topk_indices = self.self_attn(
            normed_x,
            mask,
            cache,
            prev_topk_indices,
            mask_is_causal=mask_is_causal,
        )
        h = x + r
        normed_h = _profile_stage(
            "post_attention_layernorm",
            lambda: self.post_attention_layernorm(h),
            inputs=h,
        )
        r = _profile_stage("mlp", lambda: self.mlp(normed_h), inputs=normed_h)
        output = h + r
        if profile_decode_total:
            _eval_profile_value(output)
            _record_decode_stage(
                "total_decode_layer", time.perf_counter() - total_start
            )
        return output, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.indexer_types = list(config.indexer_types or [])

    def pipeline(self, group):
        # IndexShare state is local to a stage. Keep every stage boundary on a
        # full layer so no shared layer loses the preceding layer's top-k.
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        boundaries = _index_share_pipeline_boundaries(
            self.indexer_types, self.pipeline_size
        )
        stage = self.pipeline_size - self.pipeline_rank - 1
        self.start_idx = boundaries[stage]
        self.end_idx = boundaries[stage + 1]
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = self.end_idx - self.start_idx

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        return_pre_norm_hidden: bool = False,
    ) -> mx.array:
        cache_offset = 0
        if cache is not None and cache[0] is not None:
            cache_offset = _scalar_int(cache[0][0].offset) or 0
        short_cached_decode = (
            x.shape[1] <= _MAX_QUANTIZED_SPARSE_VERIFY_TOKENS
            and cache_offset > 0
        )
        profile_token = _set_profile_scope(
            "decode"
            if x.shape[1] == 1 or short_cached_decode
            else None
        )
        profile_decode_total = profile_token is not None
        total_start = time.perf_counter() if profile_decode_total else None
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask_cache = cache[0][0] if cache[0] else None
        mask_is_causal = _has_canonical_causal_cache(mask_cache)
        mask = create_attention_mask(
            h, mask_cache, return_array=True
        )

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h,
                mask,
                cache[i],
                prev_topk_indices,
                mask_is_causal=mask_is_causal,
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        pre_norm_h = h
        h = self.norm(h)
        if profile_decode_total:
            _eval_profile_value(h)
            _record_decode_stage(
                "total_decode_model", time.perf_counter() - total_start
            )
            _GLM_DSA_PROFILE_SCOPE.reset(profile_token)
        if return_pre_norm_hidden:
            return h, pre_norm_h
        return h


class GlmMoeDsaMTPSharedHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array):
        return self.norm(x)


class GlmMoeDsaMTPPredictor(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layer = GlmMoeDsaDecoderLayer(config, config.num_hidden_layers)
        self.shared_head = GlmMoeDsaMTPSharedHead(config)

    def __call__(
        self,
        inputs_embeds: mx.array,
        previous_hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
        return_logits_hidden: bool = True,
        mask_is_causal: bool = False,
    ):
        offset = cache[0].offset if cache is not None and cache[0] is not None else 0
        if offset == 0 and inputs_embeds.shape[1] > 0:
            positions = mx.arange(inputs_embeds.shape[1]).reshape(1, -1, 1)
            inputs_embeds = mx.where(
                positions == 0,
                mx.zeros_like(inputs_embeds),
                inputs_embeds,
            )
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden = self.eh_proj(
            mx.concatenate([inputs_embeds, previous_hidden_states], axis=-1)
        )
        hidden, topk_indices = self.layer(
            hidden,
            mask,
            cache,
            prev_topk_indices,
            mask_is_causal=mask_is_causal,
        )
        logits_hidden = self.shared_head(hidden) if return_logits_hidden else None
        return hidden, logits_hidden, topk_indices


class Model(DSV32Model):
    # Prompt checkpoints contain hidden/KV states produced by the attention
    # implementation. Bump this whenever their numerical semantics change so
    # checkpoints made by a known-bad implementation are rejected on load.
    prompt_cache_semantics_version = 2

    def __init__(self, config: ModelArgs):
        nn.Module.__init__(self)
        self.args = config
        self.model_type = config.model_type
        self.model = GlmMoeDsaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = None
        if _mtp_enabled() and config.num_nextn_predict_layers > 0:
            self.mtp = GlmMoeDsaMTPPredictor(config)

    def forward_with_hidden(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out), out

    def prefill_with_hidden(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out[:, -1:, :]), out

    def sanitize(self, weights):
        mtp_layer_start = self.args.num_hidden_layers
        mtp_enabled = _mtp_enabled()

        def is_supported_internal_mtp_key(key):
            return (
                key in ("mtp.enorm.weight", "mtp.hnorm.weight", "mtp.eh_proj.weight")
                or key.startswith("mtp.shared_head.")
                or key.startswith("mtp.layer.")
            )

        def remap_key(key):
            if key.startswith("mtp."):
                if mtp_enabled and is_supported_internal_mtp_key(key):
                    return key
                return None
            if (
                key.startswith(("mtp_", "model.mtp"))
                or ".mtp." in key
                or ".mtp_" in key
            ):
                return None
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                try:
                    layer_idx = int(parts[2])
                except ValueError:
                    return key
                if layer_idx < mtp_layer_start:
                    return key
                if not mtp_enabled or layer_idx != mtp_layer_start:
                    return None
                suffix = ".".join(parts[3:])
                if suffix in ("enorm.weight", "hnorm.weight", "eh_proj.weight"):
                    return f"mtp.{suffix}"
                if suffix.startswith("shared_head."):
                    return f"mtp.{suffix}"
                if suffix:
                    return f"mtp.layer.{suffix}"
                return None
            return key

        sanitized = {}
        for k, v in weights.items():
            mapped_key = remap_key(k)
            if mapped_key is not None:
                sanitized[mapped_key] = v
        sanitized = super().sanitize(sanitized)

        if mtp_enabled:
            # The parent sanitizer handles ordinary decoder layers. Raw HF MTP
            # weights have already been remapped to ``mtp.layer`` above, so
            # apply the same expert stacking and MLA kv_b decomposition here.
            prefix = "mtp.layer"
            for projection in ("gate_proj", "down_proj", "up_proj"):
                for suffix in ("weight", "scales", "biases"):
                    first = f"{prefix}.mlp.experts.0.{projection}.{suffix}"
                    if first in sanitized:
                        to_join = [
                            sanitized.pop(
                                f"{prefix}.mlp.experts.{expert}.{projection}.{suffix}"
                            )
                            for expert in range(self.args.n_routed_experts)
                        ]
                        sanitized[
                            f"{prefix}.mlp.switch_mlp.{projection}.{suffix}"
                        ] = mx.stack(to_join)

            attention_prefix = f"{prefix}.self_attn"
            raw_kv_b = f"{attention_prefix}.kv_b_proj.weight"
            if raw_kv_b in sanitized:
                quantized = f"{attention_prefix}.kv_b_proj.scales" in sanitized
                value = sanitized.pop(raw_kv_b)
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = sanitized.pop(
                        f"{attention_prefix}.kv_b_proj.scales"
                    )
                    biases = sanitized.pop(
                        f"{attention_prefix}.kv_b_proj.biases"
                    )
                    bits = (value.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    value = mx.dequantize(
                        value,
                        scales,
                        biases,
                        bits=bits,
                        group_size=group_size,
                    )
                value = value.reshape(
                    self.args.num_attention_heads, head_dim, -1
                )
                wk = mx.contiguous(
                    value[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(value[:, self.args.qk_nope_head_dim :, :])
                if quantized:
                    wk, wk_scales, wk_biases = mx.quantize(
                        wk, bits=bits, group_size=group_size
                    )
                    wv, wv_scales, wv_biases = mx.quantize(
                        wv, bits=bits, group_size=group_size
                    )
                    sanitized[f"{attention_prefix}.embed_q.scales"] = wk_scales
                    sanitized[
                        f"{attention_prefix}.unembed_out.scales"
                    ] = wv_scales
                    sanitized[f"{attention_prefix}.embed_q.biases"] = wk_biases
                    sanitized[
                        f"{attention_prefix}.unembed_out.biases"
                    ] = wv_biases
                sanitized[f"{attention_prefix}.embed_q.weight"] = wk
                sanitized[f"{attention_prefix}.unembed_out.weight"] = wv

        return sanitized

    def make_mtp_cache(self):
        if self.mtp is None:
            return []
        if getattr(self.mtp.layer.self_attn, "skip_topk", False):
            return CacheList(GlmMlaKVCache())
        return CacheList(GlmMlaKVCache(), KVCache())

    def mtp_logits(
        self,
        inputs: mx.array,
        previous_hidden_states: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        if self.mtp is None:
            raise RuntimeError(f"GLM DSA MTP is disabled; set {GLM_DSA_MTP_ENV}=1")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(inputs)
        mask_cache = cache[0] if cache is not None else None
        mask_is_causal = (
            mask is None or (isinstance(mask, str) and mask == "causal")
        ) and _has_canonical_causal_cache(mask_cache)
        if mask is None:
            mask = create_attention_mask(
                inputs_embeds,
                cache[0] if cache is not None else None,
                return_array=True,
            )
        _hidden, logits_hidden, topk_indices = self.mtp(
            inputs_embeds,
            previous_hidden_states,
            mask,
            cache,
            prev_topk_indices,
            mask_is_causal=mask_is_causal,
        )
        return self.lm_head(logits_hidden), logits_hidden, topk_indices

    def mtp_prefill(
        self,
        inputs: mx.array,
        previous_hidden_states: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        if self.mtp is None:
            raise RuntimeError(f"GLM DSA MTP is disabled; set {GLM_DSA_MTP_ENV}=1")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(inputs)
        mask_cache = cache[0] if cache is not None else None
        mask_is_causal = (
            mask is None or (isinstance(mask, str) and mask == "causal")
        ) and _has_canonical_causal_cache(mask_cache)
        if mask is None:
            mask = create_attention_mask(
                inputs_embeds,
                cache[0] if cache is not None else None,
                return_array=True,
            )
        hidden, _logits_hidden, topk_indices = self.mtp(
            inputs_embeds,
            previous_hidden_states,
            mask,
            cache,
            prev_topk_indices,
            return_logits_hidden=False,
            mask_is_causal=mask_is_causal,
        )
        return hidden, topk_indices

    def mtp_prefill_with_last_logits(
        self,
        inputs: mx.array,
        previous_hidden_states: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        """Prefill MTP while projecting only the final draft position."""
        if self.mtp is None:
            raise RuntimeError(f"GLM DSA MTP is disabled; set {GLM_DSA_MTP_ENV}=1")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(inputs)
        mask_cache = cache[0] if cache is not None else None
        mask_is_causal = (
            mask is None or (isinstance(mask, str) and mask == "causal")
        ) and _has_canonical_causal_cache(mask_cache)
        if mask is None:
            mask = create_attention_mask(
                inputs_embeds,
                cache[0] if cache is not None else None,
                return_array=True,
            )
        hidden, _logits_hidden, topk_indices = self.mtp(
            inputs_embeds,
            previous_hidden_states,
            mask,
            cache,
            prev_topk_indices,
            return_logits_hidden=False,
            mask_is_causal=mask_is_causal,
        )
        logits_hidden = self.mtp.shared_head(hidden[:, -1:, :])
        if topk_indices is not None and topk_indices.shape[2] > 1:
            topk_indices = topk_indices[:, :, -1:, :]
        return self.lm_head(logits_hidden), logits_hidden, topk_indices

    def make_cache(self):
        # Shared layers run no indexer, so they get no indexer KVCache.
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(GlmMlaKVCache()))
            else:
                caches.append(CacheList(GlmMlaKVCache(), KVCache()))
        return caches
