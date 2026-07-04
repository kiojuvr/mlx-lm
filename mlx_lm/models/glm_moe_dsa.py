# Copyright © 2025 Apple Inc.

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx

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
GLM_DSA_NATIVE_Q8_VUP_ENV = "MLX_LM_GLM_DSA_NATIVE_Q8_VUP"
GLM_DSA_NATIVE_Q4_VUP_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_VUP"
GLM_DSA_Q_A_DENSE_CACHE_ENV = "MLX_LM_GLM_DSA_Q_A_DENSE_CACHE"
GLM_DSA_NATIVE_Q_A_RMS_NORM_ENV = "MLX_LM_GLM_DSA_NATIVE_Q_A_RMS_NORM"
GLM_DSA_NATIVE_Q4_QA_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QA"
GLM_DSA_NATIVE_Q4_QA_TILE_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QA_TILE"
GLM_DSA_NATIVE_Q4_QB_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QB"
GLM_DSA_NATIVE_Q4_QB_TILE_ENV = "MLX_LM_GLM_DSA_NATIVE_Q4_QB_TILE"
GLM_DSA_NATIVE_Q4_QB_HEAD_LAYOUT_ENV = (
    "MLX_LM_GLM_DSA_NATIVE_Q4_QB_HEAD_LAYOUT"
)
GLM_DSA_PREFILL_PROFILE_ENV = "MLX_LM_GLM_DSA_PREFILL_PROFILE"
GLM_DSA_PREFILL_PROFILE_ISOLATE_ENV = "MLX_LM_GLM_DSA_PREFILL_PROFILE_ISOLATE"

_PROFILE_STAGES = (
    "q_projection",
    "q_a_projection",
    "q_a_dense_cache_dequantization",
    "q_a_dense_projection",
    "native_q4_qa_projection",
    "q_a_layernorm",
    "native_q_a_rms_norm",
    "q_b_projection",
    "native_q4_qb_projection",
    "native_q4_qb_head_layout_projection",
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
    "total_prefill",
)
_DEFAULT_FAST_PREFILL_QUERY_CHUNK = 16
_DEFAULT_FAST_PREFILL_KEY_BLOCK = 8192
_DEFAULT_SPARSE_PREFILL_MIN_CONTEXT = 131072
_DEFAULT_NATIVE_SPARSE_PREFILL_MIN_CONTEXT = 6144
_FAST_PREFILL_LARGE_TOPK_WARNING = 1024
_LOGGER = logging.getLogger(__name__)
_GLM_DSA_PREFILL_PROFILE = None
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
_NATIVE_Q_A_RMS_NORM_LOOKUP_DONE = False
_NATIVE_Q_A_RMS_NORM_KERNEL = None
_NATIVE_Q_A_RMS_NORM_SOURCE = None
_NATIVE_Q_A_RMS_NORM_IMPORT_ERROR = None
_NATIVE_Q4_QA_LOOKUP_DONE = False
_NATIVE_Q4_QA_KERNEL = None
_NATIVE_Q4_QA_SOURCE = None
_NATIVE_Q4_QA_IMPORT_ERROR = None
_NATIVE_Q4_QB_LOOKUP_DONE = False
_NATIVE_Q4_QB_KERNEL = None
_NATIVE_Q4_QB_SOURCE = None
_NATIVE_Q4_QB_IMPORT_ERROR = None
_NATIVE_Q4_QB_HEADS_LOOKUP_DONE = False
_NATIVE_Q4_QB_HEADS_KERNEL = None
_NATIVE_Q4_QB_HEADS_SOURCE = None
_NATIVE_Q4_QB_HEADS_IMPORT_ERROR = None


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


def _native_q8_vup_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q8_VUP_ENV, False)


def _native_q4_vup_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q4_VUP_ENV, False)


def _q_a_dense_cache_enabled() -> bool:
    return _env_flag(GLM_DSA_Q_A_DENSE_CACHE_ENV, False)


def _native_q_a_rms_norm_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q_A_RMS_NORM_ENV, False)


def _native_q4_qa_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q4_QA_ENV, False)


def _native_q4_qb_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q4_QB_ENV, False)


def _native_q4_qb_head_layout_enabled() -> bool:
    return _env_flag(GLM_DSA_NATIVE_Q4_QB_HEAD_LAYOUT_ENV, False)


def _prefill_profile_enabled() -> bool:
    return _env_flag(GLM_DSA_PREFILL_PROFILE_ENV, False)


def _prefill_profile_isolate_enabled() -> bool:
    return _env_flag(GLM_DSA_PREFILL_PROFILE_ISOLATE_ENV, False)


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
        "native_q_a_rms_norm_hits": 0,
        "native_q_a_rms_norm_fallback_reasons": Counter(),
        "native_q4_qa_hits": 0,
        "native_q4_qa_fallback_reasons": Counter(),
        "native_q4_qb_hits": 0,
        "native_q4_qb_fallback_reasons": Counter(),
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
        "native_q_a_rms_norm_hits": _GLM_DSA_PREFILL_PROFILE[
            "native_q_a_rms_norm_hits"
        ],
        "native_q_a_rms_norm_fallback_reasons": dict(
            _GLM_DSA_PREFILL_PROFILE["native_q_a_rms_norm_fallback_reasons"]
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


def _record_native_q_a_rms_norm_decision(used: bool, reason: str):
    if _GLM_DSA_PREFILL_PROFILE is None:
        reset_glm_dsa_prefill_profile()
    if used:
        _GLM_DSA_PREFILL_PROFILE["native_q_a_rms_norm_hits"] += 1
    else:
        _GLM_DSA_PREFILL_PROFILE[
            "native_q_a_rms_norm_fallback_reasons"
        ][reason] += 1
    if _fast_prefill_debug_enabled():
        if used:
            _LOGGER.info(
                "GLM DSA native q_a RMSNorm enabled: source=%s",
                _NATIVE_Q_A_RMS_NORM_SOURCE or "unknown",
            )
        else:
            _LOGGER.info("GLM DSA native q_a RMSNorm fallback: %s", reason)


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


def _native_q_a_rms_norm_kernel():
    global _NATIVE_Q_A_RMS_NORM_LOOKUP_DONE
    global _NATIVE_Q_A_RMS_NORM_KERNEL
    global _NATIVE_Q_A_RMS_NORM_SOURCE
    global _NATIVE_Q_A_RMS_NORM_IMPORT_ERROR
    if _NATIVE_Q_A_RMS_NORM_LOOKUP_DONE:
        return _NATIVE_Q_A_RMS_NORM_KERNEL

    _NATIVE_Q_A_RMS_NORM_LOOKUP_DONE = True
    _NATIVE_Q_A_RMS_NORM_KERNEL = None
    _NATIVE_Q_A_RMS_NORM_SOURCE = None
    _NATIVE_Q_A_RMS_NORM_IMPORT_ERROR = None

    for module_name in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        try:
            fast = __import__(module_name, fromlist=["fast"]).fast
            has_symbol = getattr(fast, "has_symbol", None)
            if (
                has_symbol is not None
                and has_symbol("glm_dsa_q_a_rms_norm")
                and hasattr(fast, "glm_dsa_q_a_rms_norm")
            ):
                _NATIVE_Q_A_RMS_NORM_KERNEL = fast.glm_dsa_q_a_rms_norm
                _NATIVE_Q_A_RMS_NORM_SOURCE = module_name
                return _NATIVE_Q_A_RMS_NORM_KERNEL
            if _NATIVE_Q_A_RMS_NORM_IMPORT_ERROR is None and hasattr(
                fast, "import_error"
            ):
                _NATIVE_Q_A_RMS_NORM_IMPORT_ERROR = fast.import_error()
        except Exception as exc:
            if _NATIVE_Q_A_RMS_NORM_IMPORT_ERROR is None:
                _NATIVE_Q_A_RMS_NORM_IMPORT_ERROR = exc

    return _NATIVE_Q_A_RMS_NORM_KERNEL


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


def _native_q4_qb_heads_kernel():
    global _NATIVE_Q4_QB_HEADS_LOOKUP_DONE
    global _NATIVE_Q4_QB_HEADS_KERNEL
    global _NATIVE_Q4_QB_HEADS_SOURCE
    global _NATIVE_Q4_QB_HEADS_IMPORT_ERROR
    if _NATIVE_Q4_QB_HEADS_LOOKUP_DONE:
        return _NATIVE_Q4_QB_HEADS_KERNEL

    _NATIVE_Q4_QB_HEADS_LOOKUP_DONE = True
    _NATIVE_Q4_QB_HEADS_KERNEL = None
    _NATIVE_Q4_QB_HEADS_SOURCE = None
    _NATIVE_Q4_QB_HEADS_IMPORT_ERROR = None

    for module_name in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        try:
            fast = __import__(module_name, fromlist=["fast"]).fast
            has_symbol = getattr(fast, "has_symbol", None)
            if (
                has_symbol is not None
                and has_symbol("glm_dsa_q4_qb_proj_heads")
                and hasattr(fast, "glm_dsa_q4_qb_proj_heads")
            ):
                _NATIVE_Q4_QB_HEADS_KERNEL = fast.glm_dsa_q4_qb_proj_heads
                _NATIVE_Q4_QB_HEADS_SOURCE = module_name
                return _NATIVE_Q4_QB_HEADS_KERNEL
            if _NATIVE_Q4_QB_HEADS_IMPORT_ERROR is None and hasattr(
                fast, "import_error"
            ):
                _NATIVE_Q4_QB_HEADS_IMPORT_ERROR = fast.import_error()
        except Exception as exc:
            if _NATIVE_Q4_QB_HEADS_IMPORT_ERROR is None:
                _NATIVE_Q4_QB_HEADS_IMPORT_ERROR = exc

    if hasattr(mx.fast, "glm_dsa_q4_qb_proj_heads"):
        _NATIVE_Q4_QB_HEADS_KERNEL = mx.fast.glm_dsa_q4_qb_proj_heads
        _NATIVE_Q4_QB_HEADS_SOURCE = "mlx.core.fast"

    return _NATIVE_Q4_QB_HEADS_KERNEL


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
    return fast.has_symbol("dsa_indexer_scores") and fast.has_symbol(
        "dsa_topk_indices"
    )


def _native_indexer_scores(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    *,
    causal: bool,
    skip_causal_future_store: bool = False,
    causal_q_offset: int = -1,
):
    fast, _source, _error = _native_indexer_fast_module()
    if (
        fast is None
        or not fast.has_symbol("dsa_indexer_scores")
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
        or queries.dtype != weights.dtype
        or queries.dtype not in (mx.float16, mx.bfloat16)
    ):
        return None

    _B, _H, L, _D = queries.shape
    K = keys.shape[2]
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
        scores = fast.dsa_indexer_scores(
            q,
            k,
            w,
            causal=causal,
            unused_causal_prefix_topk=0,
            skip_causal_future_store=skip_causal_future_store,
            causal_q_offset=causal_q_offset,
            stream=mx.gpu,
        )
    except Exception:
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
        or not fast.has_symbol("dsa_topk_indices")
        or len(scores.shape) != 4
        or scores.shape[1] != 1
        or topk != 2048
        or scores.shape[-1] < topk
        or scores.dtype not in (mx.float16, mx.bfloat16)
    ):
        return None
    try:
        return fast.dsa_topk_indices(
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
    scores_available = fast is not None and fast.has_symbol("dsa_indexer_scores")
    topk_available = fast is not None and fast.has_symbol("dsa_topk_indices")
    return {
        "enabled": _native_indexer_enabled(),
        "available": scores_available and topk_available,
        "source": source if scores_available and topk_available else None,
        "import_error": repr(import_error) if import_error is not None else None,
        "scores_available": scores_available,
        "topk_available": topk_available,
        "min_context": 4096,
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
    heads_kernel = _native_q4_qb_heads_kernel()
    return {
        "enabled": _native_q4_qb_enabled(),
        "available": kernel is not None,
        "source": _NATIVE_Q4_QB_SOURCE,
        "import_error": (
            repr(_NATIVE_Q4_QB_IMPORT_ERROR)
            if _NATIVE_Q4_QB_IMPORT_ERROR is not None
            else None
        ),
        "head_layout_enabled": _native_q4_qb_head_layout_enabled(),
        "head_layout_available": heads_kernel is not None,
        "head_layout_source": _NATIVE_Q4_QB_HEADS_SOURCE,
        "head_layout_import_error": (
            repr(_NATIVE_Q4_QB_HEADS_IMPORT_ERROR)
            if _NATIVE_Q4_QB_HEADS_IMPORT_ERROR is not None
            else None
        ),
    }


def get_glm_dsa_native_q_a_rms_norm_status():
    kernel = _native_q_a_rms_norm_kernel()
    return {
        "enabled": _native_q_a_rms_norm_enabled(),
        "available": kernel is not None,
        "source": _NATIVE_Q_A_RMS_NORM_SOURCE,
        "import_error": (
            repr(_NATIVE_Q_A_RMS_NORM_IMPORT_ERROR)
            if _NATIVE_Q_A_RMS_NORM_IMPORT_ERROR is not None
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
    if not _prefill_profile_enabled():
        return fn()
    if inputs is not None and _prefill_profile_isolate_enabled():
        _eval_profile_value(inputs)
    start = time.perf_counter()
    value = fn()
    _eval_profile_value(value)
    _record_stage(stage, time.perf_counter() - start)
    return value


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
        object.__setattr__(self, "_q_a_dense_cache", None)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

    def _indexer_topk(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
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
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0], dtype=k.dtype))
        if k.shape[2] <= indexer.index_topk:
            return None
        native_indices = self._native_indexer_topk(q, x, k, mask)
        if native_indices is not None:
            return native_indices
        if (
            _fast_prefill_enabled()
            and b == 1
            and s > 1
            and _sparse_prefill_context_ready(k.shape[2])
        ):
            return self._block_indexer_topk(q, x, k, mask)
        return self._dense_indexer_topk(q, x, k, mask)

    def _dense_indexer_topk(
        self,
        q: mx.array,
        x: mx.array,
        k: mx.array,
        mask: Optional[mx.array],
    ):
        indexer = self.indexer
        scores = q @ k.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        weights = indexer.weights_proj(x) * (
            indexer.n_heads**-0.5 * indexer.softmax_scale
        )
        weights = weights.swapaxes(-1, -2)[..., None]
        scores = scores * weights
        scores = scores.sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-indexer.index_topk, axis=-1)[
            ..., -indexer.index_topk :
        ]

    def _native_indexer_decision(
        self,
        q: mx.array,
        x: mx.array,
        k: mx.array,
        mask: Optional[mx.array],
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
        elif mask is not None and len(mask.shape) not in (2, 4):
            return False, "mask_rank"
        B, H, L, D = q.shape
        if B != 1:
            return False, "batch_size_not_one"
        if L <= 1:
            return False, "decode"
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
    ):
        use_native, reason = self._native_indexer_decision(q, x, k, mask)
        if not use_native:
            _record_native_indexer_decision(False, reason)
            return None

        indexer = self.indexer
        weights = indexer.weights_proj(x) * (
            indexer.n_heads**-0.5 * indexer.softmax_scale
        )
        if weights.dtype != q.dtype:
            _record_native_indexer_decision(False, "mixed_weight_dtype")
            return None
        causal = mask is not None
        scores = _profile_stage(
            "native_indexer_scores",
            lambda: _native_indexer_scores(
                q,
                k,
                weights,
                causal=causal,
                skip_causal_future_store=causal,
                causal_q_offset=k.shape[2] - q.shape[2] if causal else -1,
            ),
            inputs=(q, k, weights),
        )
        if scores is None:
            _record_native_indexer_decision(False, "scores_unavailable")
            return None
        indices = _profile_stage(
            "native_indexer_topk",
            lambda: _native_indexer_topk_indices(
                scores,
                indexer.index_topk,
                bucketed=True,
                causal_valid_prefix=causal,
            ),
            inputs=scores,
        )
        if indices is None:
            _record_native_indexer_decision(False, "topk_unavailable")
            return None
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
        topk = indexer.index_topk
        query_chunk = _fast_prefill_query_chunk_size(L)
        key_block = _fast_prefill_key_block_size(total_length, topk)
        weights = indexer.weights_proj(x) * (
            indexer.n_heads**-0.5 * indexer.softmax_scale
        )
        weights = weights.swapaxes(-1, -2)[..., None]

        all_indices = []
        for q_start in range(0, L, query_chunk):
            q_stop = min(q_start + query_chunk, L)
            q_chunk = q[:, :, q_start:q_stop, :]
            weight_chunk = weights[:, :, q_start:q_stop, :]
            best_scores = None
            best_indices = None

            for k_start in range(0, total_length, key_block):
                k_stop = min(k_start + key_block, total_length)
                k_block = k[:, :, k_start:k_stop, :]
                block_scores = q_chunk @ k_block.swapaxes(-1, -2)
                block_scores = mx.maximum(block_scores, 0)
                block_scores = block_scores * weight_chunk
                block_scores = block_scores.sum(axis=1, keepdims=True)
                mask_block = _slice_attention_mask(
                    mask, q_start, q_stop, k_start, k_stop
                )
                if mask_block is not None:
                    block_scores = mx.where(mask_block, block_scores, -float("inf"))

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
        if not _sparse_prefill_context_ready(k_pe.shape[2]):
            return False, "below_sparse_min_context"
        if not _has_full_topk_causal_prefix(k_pe.shape[2], L, K):
            return False, "causal_prefix_shorter_than_topk"
        if k_pe.shape[1] != 1:
            return False, "unsupported_kv_heads"
        offset = _scalar_int(kv_cache.offset)
        if offset is None:
            return False, "non_scalar_offset"
        _warn_fast_prefill_large_topk(K)
        return True, "fast"

    def _native_sparse_prefill_decision(
        self,
        *,
        B: int,
        L: int,
        kv_cache: Any,
        kv_latent: Any,
        k_pe: mx.array,
        topk_indices: mx.array,
    ):
        if not _fast_prefill_enabled():
            return False, "fast_prefill_disabled"
        if not _native_sparse_prefill_enabled():
            return False, "disabled"
        if _native_sparse_mla_kernel() is None:
            return False, "missing_symbol"
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
        if topk_indices.shape[-1] != 2048:
            return False, f"unsupported_topk:{topk_indices.shape[-1]}"
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

    def _native_q_a_rms_norm_decision(self, x: mx.array):
        if not _native_q_a_rms_norm_enabled():
            return False, "disabled"
        if _native_q_a_rms_norm_kernel() is None:
            return False, "missing_symbol"
        weight = self.q_a_layernorm["weight"]
        if len(x.shape) != 3:
            return False, "input_rank"
        if x.shape[-1] != 2048:
            return False, f"unsupported_input_dim:{x.shape[-1]}"
        if self.q_lora_rank != 2048:
            return False, f"unsupported_q_lora_rank:{self.q_lora_rank}"
        if x.dtype not in (mx.float16, mx.bfloat16):
            return False, f"unsupported_dtype:{x.dtype}"
        if weight.dtype != x.dtype:
            return False, "mixed_dtype"
        if len(weight.shape) != 1 or weight.shape[0] != x.shape[-1]:
            return False, "weight_shape"
        return True, "native_q_a_rms_norm"

    def _q_a_layernorm(self, x: mx.array):
        use_native, reason = self._native_q_a_rms_norm_decision(x)
        if use_native:
            try:
                kernel = _native_q_a_rms_norm_kernel()
                output = _profile_stage(
                    "native_q_a_rms_norm",
                    lambda: kernel(
                        x,
                        self.q_a_layernorm["weight"],
                        float(self.q_a_layernorm.eps),
                    ),
                    inputs=x,
                )
                _record_native_q_a_rms_norm_decision(True, reason)
                return output
            except Exception as exc:
                reason = f"runtime_error:{type(exc).__name__}"

        _record_native_q_a_rms_norm_decision(False, reason)
        return self.q_a_layernorm(x)

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

    def _native_q4_qb_head_layout_decision(self, x: mx.array):
        if not _native_q4_qb_head_layout_enabled():
            return False, "head_layout_disabled"
        use_native, reason = self._native_q4_qb_decision(x)
        if not use_native:
            return False, reason
        if _native_q4_qb_heads_kernel() is None:
            return False, "missing_head_layout_symbol"
        return True, "native_q4_qb_head_layout"

    def _q_b_project_heads(self, x: mx.array):
        use_native, reason = self._native_q4_qb_head_layout_decision(x)
        if use_native:
            try:
                kernel = _native_q4_qb_heads_kernel()
                output = _profile_stage(
                    "native_q4_qb_head_layout_projection",
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

        if (
            _native_q4_qb_head_layout_enabled()
            and hasattr(self.q_b_proj, "bits")
        ):
            _record_native_q4_qb_decision(False, reason)
        return None

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
        if mask is not None:
            sparse_mask = sparse_mask & mask
        return sparse_mask

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
            q_latent = _profile_stage(
                "latent_kv_projection",
                lambda: self.embed_q(q_nope),
                inputs=q_nope,
            )
            if isinstance(kv_cache, QuantizedGlmMlaKVCache):
                kv_latent = _profile_stage(
                    "native_sparse_kv_dequantization",
                    lambda: kv_cache.dequantize_keys(kv_latent),
                    inputs=kv_latent,
                )
            output = _profile_stage(
                "native_sparse_attention",
                lambda: kernel(
                    q_latent,
                    q_pe,
                    kv_latent,
                    k_pe,
                    topk,
                    self.scale,
                    causal=True,
                ),
                inputs=(q_latent, q_pe, kv_latent, k_pe, topk),
            )
            return (
                self._unembed_out_project(output),
                (
                    "native_sparse_mla_quantized_kv"
                    if isinstance(kv_cache, QuantizedGlmMlaKVCache)
                    else "native_sparse_mla"
                ),
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
    ):
        _, _, L, _ = q_nope.shape
        query_chunk = _fast_prefill_query_chunk_size(L)
        topk = topk_indices.shape[-1]
        gather_mask = mask
        if (
            isinstance(kv_cache, (GlmMlaKVCache, QuantizedGlmMlaKVCache))
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
        if mask_selected is not None:
            pe_scores = mx.where(
                mask_selected,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )
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
            q_a = _profile_stage(
                "q_a_projection",
                lambda: self._q_a_project(x),
                inputs=x,
            )
            qr = _profile_stage(
                "q_a_layernorm",
                lambda: self._q_a_layernorm(q_a),
                inputs=q_a,
            )
            def project_q_b():
                q_heads = self._q_b_project_heads(qr)
                if q_heads is not None:
                    return q_heads
                q_flat = self._q_b_project(qr)
                return q_flat.reshape(
                    B, L, self.num_heads, self.q_head_dim
                ).transpose(0, 2, 1, 3)

            q = _profile_stage(
                "q_b_projection",
                project_q_b,
                inputs=qr,
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

        if self.indexer is not None:
            topk_indices = _profile_stage(
                "dsa_indexer_topk",
                lambda: self._indexer_topk(x, qr, mask, cache=cache[1]),
                inputs=(x, qr, mask),
            )
        else:
            topk_indices = prev_topk_indices

        fast_sparse_prefill = False
        native_sparse_prefill = False
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
                native_sparse_prefill, native_sparse_prefill_reason = (
                    self._native_sparse_prefill_decision(
                        B=B,
                        L=L,
                        kv_cache=kv_cache,
                        kv_latent=kv_latent,
                        k_pe=k_pe,
                        topk_indices=topk_indices,
                    )
                )
                if not native_sparse_prefill:
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
                if not native_sparse_prefill and not fast_sparse_prefill:
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
        if native_sparse_prefill:
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
            )
        if output is None:
            ensure_kv_latent_dequantized()
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
