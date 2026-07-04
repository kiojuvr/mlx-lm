#!/usr/bin/env python3
"""Lightweight GLM-5.2 prefill benchmark for long coding prompts."""

import argparse
import csv
import json
import logging
import math
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm.generate import PROMPT_CHECKPOINT_DEBUG_ENV, BatchGenerator, BatchStats
from mlx_lm.generate import DEFAULT_PREFILL_MAX_QK_TOKENS
from mlx_lm.generate import stream_generate
from mlx_lm.models import cache as prompt_cache
from mlx_lm.models import glm_moe_dsa
from mlx_lm.utils import load


CODING_SNIPPET = """
File: src/module_{i}.py
```python
def transform_{i}(items, config):
    result = []
    for index, item in enumerate(items):
        if config.get("enabled", True) and item.get("active", False):
            result.append((index, item["name"], item.get("value", 0)))
    return result
```

Review request:
- Find correctness issues.
- Preserve public APIs.
- Suggest focused tests.
"""


class CheckpointLogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if "prompt checkpoint:" in message:
            self.messages.append(message)


class PrefillBenchmarkStop(Exception):
    def __init__(self, processed_tokens, total_tokens):
        super().__init__(
            f"prefill stopped after {processed_tokens}/{total_tokens} tokens"
        )
        self.processed_tokens = processed_tokens
        self.total_tokens = total_tokens


def parse_lengths(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_from_q_a_kernel_sweep(value):
    allowed = {"disabled", "scaled", "wscaled", "auto"}
    candidates = [v.strip().lower() for v in value.split(",") if v.strip()]
    if not candidates:
        raise argparse.ArgumentTypeError(
            "from_q_a kernel sweep must include at least one candidate"
        )
    invalid = [candidate for candidate in candidates if candidate not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            "unsupported from_q_a kernel sweep candidate(s): "
            f"{','.join(invalid)}; expected disabled, scaled, wscaled, or auto"
        )
    return candidates


def parse_policy_candidates(value):
    candidates = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            name, length = raw.split(":", 1)
            name = name.strip()
            length = length.strip()
        else:
            name = ""
            length = raw
        length = int(length)
        if length < 0:
            raise ValueError("policy candidate lengths must be non-negative")
        candidates.append((name or f"prefix-{length}", length))
    return candidates


def safe_case_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return value or "case"


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def encode(tokenizer, text):
    return tokenizer.encode(text)


def prompt_to_tokens(tokenizer, prompt):
    if isinstance(prompt, str):
        return encode(tokenizer, prompt)
    return [int(token) for token in prompt]


def build_prompt_text(tokenizer, target_tokens, prefix_text=""):
    header = (
        "You are reviewing a large coding-agent context. "
        "Keep behavior unchanged unless the request explicitly asks otherwise.\n\n"
    )
    body = prefix_text
    i = 0
    text = header + body
    while len(encode(tokenizer, text)) < target_tokens:
        body += CODING_SNIPPET.format(i=i)
        i += 1
        text = header + body
    return text


def ds4_boundary_store_length(token_count, *, min_tokens=512, trim_tokens=32, align_tokens=2048):
    token_count = max(0, int(token_count))
    min_tokens = max(0, int(min_tokens))
    trim_tokens = max(0, int(trim_tokens))
    align_tokens = max(0, int(align_tokens))
    if token_count == 0:
        return 0
    if token_count > min_tokens + trim_tokens:
        stable_length = token_count - trim_tokens
        if align_tokens > 0:
            stable_length -= stable_length % align_tokens
        if stable_length >= min_tokens:
            return stable_length
    return token_count


def policy_sweep_candidates(args, prefix_tokens, requested_total_tokens):
    if args.policy_candidates:
        raw_candidates = args.policy_candidates
    else:
        ds4_length = min(
            prefix_tokens,
            ds4_boundary_store_length(
                requested_total_tokens,
                min_tokens=args.policy_min_tokens,
                trim_tokens=args.policy_boundary_trim_tokens,
                align_tokens=args.policy_boundary_align_tokens,
            ),
        )
        raw_candidates = [("disabled", 0), ("ds4-boundary", ds4_length)]
        if ds4_length != prefix_tokens:
            raw_candidates.append(("full-prefix", prefix_tokens))

    candidates = []
    seen = set()
    for name, store_length in raw_candidates:
        store_length = int(store_length)
        if store_length > prefix_tokens:
            raise ValueError(
                f"policy candidate {name} stores {store_length} tokens, "
                f"but shared prefix is only {prefix_tokens}"
            )
        key = (name, store_length)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((name, store_length))
    return candidates


def prefill_sweep_candidates(args):
    step_values = args.prefill_step_candidates or [512, 1024, 2048]
    qk_values = args.prefill_max_qk_token_candidates or [args.prefill_max_qk_tokens]
    adaptive_values = args.glm_dsa_adaptive_prefill_step_candidates
    min_context_values = getattr(args, "fast_prefill_min_context_candidates", None)
    if min_context_values is None:
        min_context_values = [getattr(args, "fast_prefill_min_context", None)]
    if not step_values:
        raise ValueError("prefill step candidates must not be empty")
    if not qk_values:
        raise ValueError("prefill max-qk candidates must not be empty")
    if not min_context_values:
        raise ValueError("fast prefill min-context candidates must not be empty")
    if adaptive_values is None:
        adaptive_values = [0]
        if max(step_values) < 8192:
            adaptive_values.append(8192)
    if not adaptive_values:
        raise ValueError("GLM DSA adaptive prefill step candidates must not be empty")

    candidates = []
    seen = set()
    for step_size in step_values:
        if step_size <= 0:
            raise ValueError("prefill step candidates must be positive")
        for max_qk_tokens in qk_values:
            if max_qk_tokens < 0:
                raise ValueError("prefill max-qk candidates must be non-negative")
            for adaptive_step_size in adaptive_values:
                if adaptive_step_size < 0:
                    raise ValueError(
                        "GLM DSA adaptive prefill step candidates must be non-negative"
                    )
                for min_context in min_context_values:
                    if min_context is not None and min_context < 0:
                        raise ValueError(
                            "fast prefill min-context candidates must be non-negative"
                        )
                    key = (step_size, max_qk_tokens, adaptive_step_size, min_context)
                    if key in seen:
                        continue
                    seen.add(key)
                    qk_label = (
                        "qk-off" if max_qk_tokens == 0 else f"qk-{max_qk_tokens}"
                    )
                    adaptive_label = (
                        "adaptive-off"
                        if adaptive_step_size == 0
                        else f"adaptive-{adaptive_step_size}"
                    )
                    min_context_label = (
                        "minctx-default"
                        if min_context is None
                        else f"minctx-{min_context}"
                    )
                    candidates.append(
                        {
                            "name": (
                                f"step-{step_size}-{qk_label}-{adaptive_label}-"
                                f"{min_context_label}"
                            ),
                            "prefill_step_size": step_size,
                            "prefill_max_qk_tokens": max_qk_tokens,
                            "glm_dsa_adaptive_prefill_step_size": adaptive_step_size,
                            "fast_prefill_min_context": min_context,
                        }
                    )
    return candidates


def build_prompt_tokens(tokenizer, target_tokens, prefix_text=""):
    return encode(tokenizer, build_prompt_text(tokenizer, target_tokens, prefix_text))[
        :target_tokens
    ]


def build_queued_prompt_texts(tokenizer, args):
    prefix = build_prompt_text(tokenizer, args.queued_prefix_tokens)
    suffix_lengths = parse_lengths(args.queued_suffix_tokens)
    prompts = []
    for i in range(args.queued_requests):
        suffix_target = suffix_lengths[i % len(suffix_lengths)]
        target_tokens = args.queued_prefix_tokens + suffix_target
        text = (
            f"{prefix}\n\n"
            f"Queued coding-agent request {i}:\n"
            "Review only the changed suffix and keep the shared context in mind.\n"
        )
        snippet_id = i * 100
        while len(encode(tokenizer, text)) < target_tokens:
            text += CODING_SNIPPET.format(i=snippet_id)
            snippet_id += 1
        prompts.append(text)
    return prompts


def parse_checkpoint_log_value(value):
    if value == "None":
        return None
    if value.startswith("{") or value.startswith("["):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def chunk_sparse_route(values):
    if int(values.get("glm_dsa_native_sparse_prefill_hits", 0) or 0) > 0:
        return "native_sparse"
    if int(values.get("glm_dsa_fast_prefill_hits", 0) or 0) > 0:
        return "fast_sparse"
    return "dense"


def extract_checkpoint_summary(messages):
    summary = {
        "checkpoint_total_prompt_tokens": None,
        "server_cached_tokens": None,
        "disk_cached_tokens": None,
        "fresh_prompt_tokens": None,
        "fresh_prefill_tokens": None,
        "checkpoint_prefill_step_size": None,
        "checkpoint_prefill_max_qk_tokens": None,
        "checkpoint_glm_dsa_adaptive_prefill_step_size": None,
        "checkpoint_glm_dsa_adaptive_prefill_after_tokens": None,
        "checkpoint_glm_dsa_adaptive_prefill_min_remaining_tokens": None,
        "checkpoint_prefill_chunks": 0,
        "checkpoint_max_adaptive_prefill_step_size": None,
        "checkpoint_max_effective_prefill_step_size": None,
        "checkpoint_prefill_chunk_seconds_total": 0.0,
        "checkpoint_max_prefill_chunk_seconds": None,
        "checkpoint_slowest_prefill_chunk_start_tokens": None,
        "checkpoint_slowest_prefill_chunk_tokens": None,
        "checkpoint_slowest_prefill_chunk_route": None,
        "checkpoint_native_sparse_prefill_chunks": 0,
        "checkpoint_fast_sparse_prefill_chunks": 0,
        "checkpoint_dense_prefill_chunks": 0,
        "checkpoint_native_indexer_chunks": 0,
        "checkpoint_prefill_chunk_summaries": [],
        "checkpoint_resolution": None,
        "checkpoint_lookup_seconds": None,
        "checkpoint_files_scanned": None,
        "checkpoint_candidates_scanned": None,
        "checkpoint_matched_candidates": None,
        "checkpoint_manifest_entries": None,
        "checkpoint_manifest_bootstrap": None,
        "checkpoint_events": len(messages),
    }
    for message in messages:
        if "prefill summary " in message:
            for key, value in re.findall(r"([a-z0-9_]+)=([^ ]+)", message):
                output_key = {
                    "total_prompt_tokens": "checkpoint_total_prompt_tokens",
                    "prefill_step_size": "checkpoint_prefill_step_size",
                    "prefill_max_qk_tokens": "checkpoint_prefill_max_qk_tokens",
                    "glm_dsa_adaptive_prefill_step_size": (
                        "checkpoint_glm_dsa_adaptive_prefill_step_size"
                    ),
                    "glm_dsa_adaptive_prefill_after_tokens": (
                        "checkpoint_glm_dsa_adaptive_prefill_after_tokens"
                    ),
                    "glm_dsa_adaptive_prefill_min_remaining_tokens": (
                        "checkpoint_glm_dsa_adaptive_prefill_min_remaining_tokens"
                    ),
                    "files_scanned": "checkpoint_files_scanned",
                    "candidates_scanned": "checkpoint_candidates_scanned",
                    "matched_candidates": "checkpoint_matched_candidates",
                    "manifest_entries": "checkpoint_manifest_entries",
                    "manifest_bootstrap": "checkpoint_manifest_bootstrap",
                    "lookup_seconds": "checkpoint_lookup_seconds",
                    "resolution": "checkpoint_resolution",
                }.get(key, key)
                if output_key not in summary:
                    continue
                if output_key == "checkpoint_resolution":
                    summary[output_key] = value
                elif output_key == "checkpoint_lookup_seconds":
                    summary[output_key] = float(value)
                elif value == "None":
                    summary[output_key] = None
                else:
                    summary[output_key] = int(value)
            continue

        if "prefill chunk " not in message:
            continue
        values = {
            key: parse_checkpoint_log_value(value)
            for key, value in re.findall(r"([a-z0-9_]+)=([^ ]+)", message)
        }
        summary["checkpoint_prefill_chunks"] += 1
        route = chunk_sparse_route(values)
        if route == "native_sparse":
            summary["checkpoint_native_sparse_prefill_chunks"] += 1
        elif route == "fast_sparse":
            summary["checkpoint_fast_sparse_prefill_chunks"] += 1
        else:
            summary["checkpoint_dense_prefill_chunks"] += 1
        if int(values.get("glm_dsa_native_indexer_hits", 0) or 0) > 0:
            summary["checkpoint_native_indexer_chunks"] += 1
        chunk_seconds = values.get("chunk_seconds")
        if chunk_seconds is not None:
            chunk_seconds = float(chunk_seconds)
            summary["checkpoint_prefill_chunk_seconds_total"] += chunk_seconds
            if (
                summary["checkpoint_max_prefill_chunk_seconds"] is None
                or chunk_seconds > summary["checkpoint_max_prefill_chunk_seconds"]
            ):
                summary["checkpoint_max_prefill_chunk_seconds"] = chunk_seconds
                summary["checkpoint_slowest_prefill_chunk_start_tokens"] = values.get(
                    "start_tokens"
                )
                summary["checkpoint_slowest_prefill_chunk_tokens"] = values.get(
                    "chunk_tokens"
                )
                summary["checkpoint_slowest_prefill_chunk_route"] = route
        for key, output_key in (
            (
                "adaptive_prefill_step_size",
                "checkpoint_max_adaptive_prefill_step_size",
            ),
            (
                "effective_prefill_step_size",
                "checkpoint_max_effective_prefill_step_size",
            ),
        ):
            if key not in values:
                continue
            value = int(values[key])
            current = summary[output_key]
            summary[output_key] = value if current is None else max(current, value)
        chunk_summary = {
            "start_tokens": values.get("start_tokens"),
            "chunk_tokens": values.get("chunk_tokens"),
            "processed_tokens": values.get("processed_tokens"),
            "route": route,
            "chunk_seconds": chunk_seconds,
            "native_sparse_prefill_hits": values.get(
                "glm_dsa_native_sparse_prefill_hits",
                0,
            ),
            "fast_prefill_hits": values.get("glm_dsa_fast_prefill_hits", 0),
            "native_indexer_hits": values.get("glm_dsa_native_indexer_hits", 0),
            "native_sparse_prefill_fallback_reasons": values.get(
                "glm_dsa_native_sparse_prefill_fallback_reasons",
                {},
            ),
            "fast_prefill_fallback_reasons": values.get(
                "glm_dsa_fallback_reasons",
                {},
            ),
            "native_indexer_fallback_reasons": values.get(
                "glm_dsa_native_indexer_fallback_reasons",
                {},
            ),
            "native_indexer_scores_seconds": values.get(
                "glm_dsa_native_indexer_scores_seconds"
            ),
            "native_indexer_topk_seconds": values.get(
                "glm_dsa_native_indexer_topk_seconds"
            ),
            "native_sparse_attention_seconds": values.get(
                "glm_dsa_native_sparse_attention_seconds"
            ),
            "attention_seconds": values.get("glm_dsa_attention_seconds"),
            "latent_kv_projection_seconds": values.get(
                "glm_dsa_latent_kv_projection_seconds"
            ),
        }
        summary["checkpoint_prefill_chunk_summaries"].append(chunk_summary)
    return summary


def effective_native_q4_qa_tile() -> str:
    value = os.environ.get(glm_moe_dsa.GLM_DSA_NATIVE_Q4_QA_TILE_ENV, "default")
    tile = value.strip().lower()
    if tile in ("", "default"):
        return "bk64"
    return tile


def effective_native_q4_qb_tile() -> str:
    value = os.environ.get(glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_TILE_ENV, "default")
    tile = value.strip().lower()
    if tile in ("", "default"):
        return "bm64"
    return tile


def effective_native_q4_qb_scaled_tile() -> str:
    value = os.environ.get(
        glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_SCALED_TILE_ENV, "default"
    )
    tile = value.strip().lower()
    if tile in ("", "default"):
        return "bn64"
    return tile


def effective_sparse_mla_tile() -> str:
    value = os.environ.get(glm_moe_dsa.GLM_DSA_SPARSE_MLA_TILE_ENV, "default")
    tile = value.strip().lower()
    if tile in ("", "default"):
        return "bk256_dc32_wm8"
    return tile


def configure_glm_dsa_fast_prefill(args):
    if args.fast_prefill == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_FAST_PREFILL_ENV] = "1"
    elif args.fast_prefill == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_FAST_PREFILL_ENV] = "0"
    native_sparse_prefill = getattr(args, "native_sparse_prefill", "default")
    if native_sparse_prefill == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_ENV] = "1"
    elif native_sparse_prefill == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_ENV] = "0"
    native_sparse_quantized_kv = getattr(
        args, "native_sparse_quantized_kv", "default"
    )
    if native_sparse_quantized_kv == "enabled":
        os.environ[
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV
        ] = "1"
    elif native_sparse_quantized_kv == "disabled":
        os.environ[
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV
        ] = "0"
    native_sparse_quantized_kv_max_context = getattr(
        args,
        "native_sparse_quantized_kv_max_context",
        None,
    )
    if native_sparse_quantized_kv_max_context is not None:
        os.environ[
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT_ENV
        ] = str(native_sparse_quantized_kv_max_context)
    native_indexer = getattr(args, "native_indexer", "default")
    if native_indexer == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_INDEXER_ENV] = "1"
    elif native_indexer == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_INDEXER_ENV] = "0"
    native_q8_vup = getattr(args, "native_q8_vup", "default")
    if native_q8_vup == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q8_VUP_ENV] = "1"
    elif native_q8_vup == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q8_VUP_ENV] = "0"
    native_q4_vup = getattr(args, "native_q4_vup", "default")
    if native_q4_vup == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_VUP_ENV] = "1"
    elif native_q4_vup == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_VUP_ENV] = "0"
    q_a_dense_cache = getattr(args, "q_a_dense_cache", "default")
    if q_a_dense_cache == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_Q_A_DENSE_CACHE_ENV] = "1"
    elif q_a_dense_cache == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_Q_A_DENSE_CACHE_ENV] = "0"
    native_q_a_rms_norm = getattr(args, "native_q_a_rms_norm", "default")
    if native_q_a_rms_norm == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q_A_RMS_NORM_ENV] = "1"
    elif native_q_a_rms_norm == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q_A_RMS_NORM_ENV] = "0"
    native_q4_qa = getattr(args, "native_q4_qa", "default")
    if native_q4_qa == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QA_ENV] = "1"
    elif native_q4_qa == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QA_ENV] = "0"
    native_q4_qa_tile = getattr(args, "native_q4_qa_tile", "default")
    if native_q4_qa_tile != "default":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QA_TILE_ENV] = native_q4_qa_tile
    native_sparse_mla_tile = getattr(args, "native_sparse_mla_tile", "default")
    if native_sparse_mla_tile != "default":
        os.environ[
            glm_moe_dsa.GLM_DSA_SPARSE_MLA_TILE_ENV
        ] = native_sparse_mla_tile
    native_q4_qb = getattr(args, "native_q4_qb", "default")
    if native_q4_qb == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_ENV] = "1"
    elif native_q4_qb == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_ENV] = "0"
    native_q4_qb_tile = getattr(args, "native_q4_qb_tile", "default")
    if native_q4_qb_tile != "default":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_TILE_ENV] = native_q4_qb_tile
    native_q4_qb_scaled_tile = getattr(
        args, "native_q4_qb_scaled_tile", "default"
    )
    if native_q4_qb_scaled_tile != "default":
        os.environ[
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_SCALED_TILE_ENV
        ] = native_q4_qb_scaled_tile
    native_q4_qb_head_layout = getattr(
        args, "native_q4_qb_head_layout", "default"
    )
    if native_q4_qb_head_layout == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_HEAD_LAYOUT_ENV] = "1"
    elif native_q4_qb_head_layout == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_HEAD_LAYOUT_ENV] = "0"
    native_q4_qb_from_q_a = getattr(args, "native_q4_qb_from_q_a", "default")
    if native_q4_qb_from_q_a == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV] = "1"
    elif native_q4_qb_from_q_a == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV] = "0"
    native_q4_qb_from_q_a_kernel = getattr(
        args,
        "native_q4_qb_from_q_a_kernel",
        "default",
    )
    if native_q4_qb_from_q_a_kernel != "default":
        os.environ[
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV
        ] = native_q4_qb_from_q_a_kernel
    if args.fast_prefill_query_chunk is not None:
        os.environ[glm_moe_dsa.GLM_DSA_FAST_PREFILL_QUERY_CHUNK_ENV] = str(
            args.fast_prefill_query_chunk
        )
    if args.fast_prefill_key_block is not None:
        os.environ[glm_moe_dsa.GLM_DSA_FAST_PREFILL_KEY_BLOCK_ENV] = str(
            args.fast_prefill_key_block
        )
    if args.fast_prefill_min_context is not None:
        os.environ[glm_moe_dsa.GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV] = str(
            args.fast_prefill_min_context
        )
    native_sparse_prefill_min_context = getattr(
        args, "native_sparse_prefill_min_context", None
    )
    if native_sparse_prefill_min_context is not None:
        os.environ[glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT_ENV] = str(
            native_sparse_prefill_min_context
        )
    if args.prefill_profile:
        os.environ[glm_moe_dsa.GLM_DSA_PREFILL_PROFILE_ENV] = "1"
    prefill_profile_isolate = getattr(args, "prefill_profile_isolate", "default")
    if prefill_profile_isolate == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_PREFILL_PROFILE_ISOLATE_ENV] = "1"
    elif prefill_profile_isolate == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_PREFILL_PROFILE_ISOLATE_ENV] = "0"


def reset_glm_dsa_profile():
    glm_moe_dsa.reset_glm_dsa_prefill_profile()


def collect_glm_dsa_profile(args):
    profile = glm_moe_dsa.get_glm_dsa_prefill_profile()
    native_status = glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status()
    native_indexer_status = glm_moe_dsa.get_glm_dsa_native_indexer_status()
    native_q8_vup_status = glm_moe_dsa.get_glm_dsa_native_q8_vup_status()
    native_q4_vup_status = glm_moe_dsa.get_glm_dsa_native_q4_vup_status()
    native_q_a_rms_norm_status = (
        glm_moe_dsa.get_glm_dsa_native_q_a_rms_norm_status()
    )
    native_q4_qa_status = glm_moe_dsa.get_glm_dsa_native_q4_qa_status()
    native_q4_qb_status = glm_moe_dsa.get_glm_dsa_native_q4_qb_status()
    stage_values = {}
    for stage, values in profile["stages"].items():
        key = f"glm_dsa_{stage}_seconds"
        stage_values[key] = values["seconds"] if args.prefill_profile else None
        stage_values[f"glm_dsa_{stage}_count"] = values["count"]
    return {
        "glm_dsa_prefill_profile": bool(args.prefill_profile),
        "glm_dsa_prefill_profile_isolate": getattr(
            args, "prefill_profile_isolate", "default"
        ),
        "glm_dsa_prefill_profile_isolate_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_PREFILL_PROFILE_ISOLATE_ENV,
            "default-off",
        ),
        "glm_dsa_fast_prefill": args.fast_prefill,
        "glm_dsa_fast_prefill_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_FAST_PREFILL_ENV,
            "default-on",
        ),
        "glm_dsa_fast_prefill_query_chunk": os.environ.get(
            glm_moe_dsa.GLM_DSA_FAST_PREFILL_QUERY_CHUNK_ENV,
            "default",
        ),
        "glm_dsa_fast_prefill_key_block": os.environ.get(
            glm_moe_dsa.GLM_DSA_FAST_PREFILL_KEY_BLOCK_ENV,
            "default",
        ),
        "glm_dsa_sparse_prefill_min_context": os.environ.get(
            glm_moe_dsa.GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV,
            "default",
        ),
        "glm_dsa_native_sparse_prefill": getattr(
            args, "native_sparse_prefill", "default"
        ),
        "glm_dsa_native_sparse_prefill_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_ENV,
            "default-on",
        ),
        "glm_dsa_native_sparse_prefill_available": native_status["available"],
        "glm_dsa_native_sparse_prefill_source": native_status["source"],
        "glm_dsa_native_sparse_prefill_import_error": native_status["import_error"],
        "glm_dsa_native_sparse_prefill_min_context": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_MIN_CONTEXT_ENV,
            "default",
        ),
        "glm_dsa_native_sparse_prefill_quantized_kv": getattr(
            args, "native_sparse_quantized_kv", "default"
        ),
        "glm_dsa_native_sparse_prefill_quantized_kv_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV,
            "default-off",
        ),
        "glm_dsa_native_sparse_prefill_quantized_kv_max_context": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT_ENV,
            "default",
        ),
        "glm_dsa_sparse_mla_tile_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_SPARSE_MLA_TILE_ENV,
            "default",
        ),
        "glm_dsa_sparse_mla_tile": effective_sparse_mla_tile(),
        "glm_dsa_fast_prefill_hits": profile["fast_prefill_hits"],
        "glm_dsa_fast_prefill_fallback_reasons": profile["fallback_reasons"],
        "glm_dsa_native_sparse_prefill_hits": profile[
            "native_sparse_prefill_hits"
        ],
        "glm_dsa_native_sparse_prefill_fallback_reasons": profile[
            "native_sparse_prefill_fallback_reasons"
        ],
        "glm_dsa_native_indexer": getattr(args, "native_indexer", "default"),
        "glm_dsa_native_indexer_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_INDEXER_ENV,
            "default-on",
        ),
        "glm_dsa_native_indexer_available": native_indexer_status["available"],
        "glm_dsa_native_indexer_source": native_indexer_status["source"],
        "glm_dsa_native_indexer_import_error": native_indexer_status["import_error"],
        "glm_dsa_native_indexer_scores_available": native_indexer_status[
            "scores_available"
        ],
        "glm_dsa_native_indexer_topk_available": native_indexer_status[
            "topk_available"
        ],
        "glm_dsa_native_indexer_min_context": native_indexer_status["min_context"],
        "glm_dsa_native_indexer_hits": profile.get("native_indexer_hits", 0),
        "glm_dsa_native_indexer_fallback_reasons": profile.get(
            "native_indexer_fallback_reasons", {}
        ),
        "glm_dsa_native_q8_vup_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q8_VUP_ENV,
            "default-off",
        ),
        "glm_dsa_native_q8_vup_available": native_q8_vup_status["available"],
        "glm_dsa_native_q8_vup_source": native_q8_vup_status["source"],
        "glm_dsa_native_q8_vup_import_error": native_q8_vup_status["import_error"],
        "glm_dsa_native_q8_vup_hits": profile["native_q8_vup_hits"],
        "glm_dsa_native_q8_vup_fallback_reasons": profile[
            "native_q8_vup_fallback_reasons"
        ],
        "glm_dsa_native_q4_vup": getattr(args, "native_q4_vup", "default"),
        "glm_dsa_native_q4_vup_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_VUP_ENV,
            "default-off",
        ),
        "glm_dsa_native_q4_vup_available": native_q4_vup_status["available"],
        "glm_dsa_native_q4_vup_source": native_q4_vup_status["source"],
        "glm_dsa_native_q4_vup_import_error": native_q4_vup_status[
            "import_error"
        ],
        "glm_dsa_native_q4_vup_hits": profile["native_q4_vup_hits"],
        "glm_dsa_native_q4_vup_fallback_reasons": profile[
            "native_q4_vup_fallback_reasons"
        ],
        "glm_dsa_q_a_dense_cache": getattr(args, "q_a_dense_cache", "default"),
        "glm_dsa_q_a_dense_cache_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_Q_A_DENSE_CACHE_ENV,
            "default-off",
        ),
        "glm_dsa_q_a_dense_cache_hits": profile["q_a_dense_cache_hits"],
        "glm_dsa_q_a_dense_cache_builds": profile["q_a_dense_cache_builds"],
        "glm_dsa_q_a_dense_cache_fallback_reasons": profile[
            "q_a_dense_cache_fallback_reasons"
        ],
        "glm_dsa_native_q_a_rms_norm": getattr(
            args, "native_q_a_rms_norm", "default"
        ),
        "glm_dsa_native_q_a_rms_norm_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q_A_RMS_NORM_ENV,
            "default-off",
        ),
        "glm_dsa_native_q_a_rms_norm_available": native_q_a_rms_norm_status[
            "available"
        ],
        "glm_dsa_native_q_a_rms_norm_source": native_q_a_rms_norm_status[
            "source"
        ],
        "glm_dsa_native_q_a_rms_norm_import_error": native_q_a_rms_norm_status[
            "import_error"
        ],
        "glm_dsa_native_q_a_rms_norm_hits": profile.get(
            "native_q_a_rms_norm_hits", 0
        ),
        "glm_dsa_native_q_a_rms_norm_fallback_reasons": profile.get(
            "native_q_a_rms_norm_fallback_reasons", {}
        ),
        "glm_dsa_native_q4_qa": getattr(args, "native_q4_qa", "default"),
        "glm_dsa_native_q4_qa_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QA_ENV,
            "default-off",
        ),
        "glm_dsa_native_q4_qa_tile_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QA_TILE_ENV,
            "default",
        ),
        "glm_dsa_native_q4_qa_tile": effective_native_q4_qa_tile(),
        "glm_dsa_native_q4_qa_available": native_q4_qa_status["available"],
        "glm_dsa_native_q4_qa_source": native_q4_qa_status["source"],
        "glm_dsa_native_q4_qa_import_error": native_q4_qa_status["import_error"],
        "glm_dsa_native_q4_qa_hits": profile["native_q4_qa_hits"],
        "glm_dsa_native_q4_qa_fallback_reasons": profile[
            "native_q4_qa_fallback_reasons"
        ],
        "glm_dsa_native_q4_qb": getattr(args, "native_q4_qb", "default"),
        "glm_dsa_native_q4_qb_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_ENV,
            "default-off",
        ),
        "glm_dsa_native_q4_qb_tile_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_TILE_ENV,
            "default",
        ),
        "glm_dsa_native_q4_qb_tile": effective_native_q4_qb_tile(),
        "glm_dsa_native_q4_qb_scaled_tile_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_SCALED_TILE_ENV,
            "default",
        ),
        "glm_dsa_native_q4_qb_scaled_tile": effective_native_q4_qb_scaled_tile(),
        "glm_dsa_native_q4_qb_head_layout": getattr(
            args, "native_q4_qb_head_layout", "default"
        ),
        "glm_dsa_native_q4_qb_head_layout_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_HEAD_LAYOUT_ENV,
            "default-off",
        ),
        "glm_dsa_native_q4_qb_head_layout_available": native_q4_qb_status.get(
            "head_layout_available", False
        ),
        "glm_dsa_native_q4_qb_head_layout_source": native_q4_qb_status.get(
            "head_layout_source"
        ),
        "glm_dsa_native_q4_qb_head_layout_import_error": native_q4_qb_status.get(
            "head_layout_import_error"
        ),
        "glm_dsa_native_q4_qb_from_q_a": getattr(
            args, "native_q4_qb_from_q_a", "default"
        ),
        "glm_dsa_native_q4_qb_from_q_a_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV,
            "default-off",
        ),
        "glm_dsa_native_q4_qb_from_q_a_kernel": native_q4_qb_status.get(
            "from_q_a_kernel"
        ),
        "glm_dsa_native_q4_qb_from_q_a_kernel_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV,
            "default-scaled",
        ),
        "glm_dsa_native_q4_qb_from_q_a_available": native_q4_qb_status.get(
            "from_q_a_available", False
        ),
        "glm_dsa_native_q4_qb_from_q_a_rms_scale_source": native_q4_qb_status.get(
            "from_q_a_rms_scale_source"
        ),
        "glm_dsa_native_q4_qb_from_q_a_projection_source": native_q4_qb_status.get(
            "from_q_a_projection_source"
        ),
        "glm_dsa_native_q4_qb_from_q_a_wscaled_heads_source": native_q4_qb_status.get(
            "from_q_a_wscaled_heads_source"
        ),
        "glm_dsa_native_q4_qb_from_q_a_scaled_heads_source": native_q4_qb_status.get(
            "from_q_a_scaled_heads_source"
        ),
        "glm_dsa_native_q4_qb_from_q_a_import_error": native_q4_qb_status.get(
            "from_q_a_import_error"
        ),
        "glm_dsa_native_q4_qb_available": native_q4_qb_status["available"],
        "glm_dsa_native_q4_qb_source": native_q4_qb_status["source"],
        "glm_dsa_native_q4_qb_import_error": native_q4_qb_status["import_error"],
        "glm_dsa_native_q4_qb_hits": profile["native_q4_qb_hits"],
        "glm_dsa_native_q4_qb_fallback_reasons": profile[
            "native_q4_qb_fallback_reasons"
        ],
        "glm_dsa_native_q4_qb_from_q_a_hits": profile.get(
            "native_q4_qb_from_q_a_hits", 0
        ),
        "glm_dsa_native_q4_qb_from_q_a_fallback_reasons": profile.get(
            "native_q4_qb_from_q_a_fallback_reasons", {}
        ),
        **native_sparse_prefill_route_diagnostics(args, profile, native_status),
        "glm_dsa_native_q8_vup": getattr(args, "native_q8_vup", "default"),
        **stage_values,
    }


def primary_counter_reason(reasons):
    if not reasons:
        return None
    return sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _fast_prefill_disabled_by_config(args):
    if getattr(args, "fast_prefill", "default") == "disabled":
        return True
    value = os.environ.get(glm_moe_dsa.GLM_DSA_FAST_PREFILL_ENV)
    if value is None:
        return False
    return value.strip().lower() in ("", "0", "false", "no", "off")


def native_sparse_prefill_attempt_min_context(native_status):
    return int(native_status.get("min_context") or 0)


def native_sparse_prefill_config_blocker(args, native_status):
    if _fast_prefill_disabled_by_config(args):
        return "fast_prefill_disabled"
    if not native_status["enabled"]:
        return "disabled"
    if not native_status["available"]:
        return "missing_symbol"
    mode = getattr(args, "mode", "single")
    batch_size = getattr(args, "batch_size", 1)
    if mode == "batch" and batch_size != 1:
        return "batch_size_not_one"
    kv_bits = getattr(args, "kv_bits", None)
    if kv_bits is None:
        return None
    if kv_bits != 8:
        return f"unsupported_kv_bits:{kv_bits}"
    quantized_kv_enabled = os.environ.get(
        glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV
    )
    quantized_kv_opt_in = (
        quantized_kv_enabled is not None
        and quantized_kv_enabled.strip().lower()
        not in ("", "0", "false", "no", "off")
    )
    if quantized_kv_opt_in:
        return None
    attempt_min_context = native_sparse_prefill_attempt_min_context(native_status)
    quantized_kv_start = getattr(args, "quantized_kv_start", 0)
    if quantized_kv_start <= attempt_min_context:
        return "quantized_kv_at_native_threshold"
    return "quantized_kv_after_native_threshold"


def native_sparse_prefill_route_state(profile, native_status):
    hits = profile["native_sparse_prefill_hits"]
    fallback_reasons = profile["native_sparse_prefill_fallback_reasons"]
    if hits and fallback_reasons:
        return "hit_with_fallbacks"
    if hits:
        return "hit"
    if fallback_reasons:
        return "fallback"
    if not native_status["enabled"]:
        return "disabled"
    if not native_status["available"]:
        return "unavailable"
    return "not_attempted"


def native_sparse_prefill_route_diagnostics(args, profile, native_status):
    return {
        "glm_dsa_native_sparse_prefill_route_state": (
            native_sparse_prefill_route_state(profile, native_status)
        ),
        "glm_dsa_native_sparse_prefill_primary_fallback": primary_counter_reason(
            profile["native_sparse_prefill_fallback_reasons"]
        ),
        "glm_dsa_native_sparse_prefill_config_blocker": (
            native_sparse_prefill_config_blocker(args, native_status)
        ),
        "glm_dsa_native_sparse_prefill_attempt_min_context": (
            native_sparse_prefill_attempt_min_context(native_status)
        ),
    }


def _native_smoke_status_fields(args, status):
    return {
        "glm_dsa_native_sparse_prefill": getattr(
            args, "native_sparse_prefill", "default"
        ),
        "glm_dsa_native_sparse_prefill_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_ENV,
            "default-on",
        ),
        "glm_dsa_native_sparse_prefill_available": status["available"],
        "glm_dsa_native_sparse_prefill_source": status["source"],
        "glm_dsa_native_sparse_prefill_import_error": status["import_error"],
        "glm_dsa_native_sparse_prefill_min_context": status["min_context"],
        "glm_dsa_native_sparse_prefill_quantized_kv": getattr(
            args, "native_sparse_quantized_kv", "default"
        ),
        "glm_dsa_native_sparse_prefill_quantized_kv_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_ENV,
            "default-off",
        ),
        "glm_dsa_native_sparse_prefill_quantized_kv_max_context": os.environ.get(
            glm_moe_dsa.GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV_MAX_CONTEXT_ENV,
            "default",
        ),
        "glm_dsa_sparse_mla_tile_env": os.environ.get(
            glm_moe_dsa.GLM_DSA_SPARSE_MLA_TILE_ENV,
            "default",
        ),
        "glm_dsa_sparse_mla_tile": effective_sparse_mla_tile(),
    }


def _native_smoke_symbols(source):
    if source not in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        if source == "mlx.core.fast":
            return ("glm_dsa_sparse_mla_attention",)
        return ()
    fast = __import__(source, fromlist=["fast"]).fast
    native_symbols = getattr(fast, "native_symbols", None)
    if native_symbols is None:
        return ()
    return tuple(native_symbols())


def _native_smoke_kernel(source):
    if source in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        fast = __import__(source, fromlist=["fast"]).fast
        return fast.glm_dsa_sparse_mla_attention
    if source == "mlx.core.fast":
        return mx.fast.glm_dsa_sparse_mla_attention
    loader = getattr(glm_moe_dsa, "_native_sparse_mla_kernel", None)
    if loader is None:
        return None
    return loader()


def _native_q8_vup_smoke_kernel(source):
    if source in (
        "mlx_lm.custom_kernels.glm_moe_dsa",
        "omlx.custom_kernels.glm_moe_dsa",
    ):
        fast = __import__(source, fromlist=["fast"]).fast
        return fast.glm_dsa_q8_vup_flat
    if source == "mlx.core.fast":
        return mx.fast.glm_dsa_q8_vup_flat
    loader = getattr(glm_moe_dsa, "_native_q8_vup_kernel", None)
    if loader is None:
        return None
    return loader()


def _native_smoke_dense_reference(q_latent, q_pe, kv_latent, k_pe, scale):
    B, H, L, _ = q_latent.shape
    K = kv_latent.shape[2]
    latent = mx.broadcast_to(kv_latent, (B, H, K, kv_latent.shape[-1]))
    pe = mx.broadcast_to(k_pe, (B, H, K, k_pe.shape[-1]))
    scores = mx.matmul(q_latent, mx.swapaxes(latent, -1, -2))
    scores = scores + mx.matmul(q_pe, mx.swapaxes(pe, -1, -2))
    scores = scores * scale

    q_positions = mx.arange(L).reshape(1, 1, L, 1)
    k_positions = mx.arange(K).reshape(1, 1, 1, K)
    causal_mask = k_positions <= (K - L + q_positions)
    scores = mx.where(causal_mask, scores, mx.array(-1e9, dtype=scores.dtype))
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q_latent.dtype)
    return mx.matmul(weights, latent)


def benchmark_mx_callable(fn, runs, warmup_runs):
    for _ in range(warmup_runs):
        value = fn()
        mx.eval(value)
    mx.synchronize()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        value = fn()
        mx.eval(value)
        mx.synchronize()
        timings.append(time.perf_counter() - start)
    return {
        "mean": statistics.mean(timings),
        "min": min(timings),
        "p50": percentile(timings, 50),
    }


def _run_native_q8_vup_smoke(row, args, status):
    if not status["available"]:
        row["native_q8_vup_smoke_error"] = "native q8 V-up kernel unavailable"
        return
    try:
        kernel = _native_q8_vup_smoke_kernel(status["source"])
        if kernel is None:
            raise RuntimeError("native q8 V-up kernel unavailable")

        B, H = 1, 64
        L = args.native_smoke_q_len
        latent_dim = 512
        value_dim = 256
        mx.random.seed(args.native_smoke_seed + 1)
        x = mx.random.normal((B, H, L, latent_dim), dtype=mx.float16) * 0.02
        weight = mx.random.normal(
            (H, value_dim, latent_dim), dtype=mx.float16
        ) * 0.02
        q_weight, scales, biases = mx.quantize(
            weight,
            group_size=64,
            bits=8,
            mode="affine",
        )
        output = kernel(x, q_weight, scales, biases)
        reference = _native_q8_vup_reference(x, q_weight, scales, biases)
        diff = mx.abs(output.astype(mx.float32) - reference.astype(mx.float32))
        mx.eval(output, reference, diff)

        max_abs_diff = float(mx.max(diff).item())
        mean_abs_diff = float(mx.mean(diff).item())
        row.update(
            {
                "native_q8_vup_smoke_shape": list(output.shape),
                "native_q8_vup_smoke_max_abs_diff": max_abs_diff,
                "native_q8_vup_smoke_mean_abs_diff": mean_abs_diff,
                "native_q8_vup_smoke_passed": (
                    max_abs_diff <= args.native_smoke_max_diff
                ),
            }
        )
        benchmark_runs = getattr(args, "native_smoke_benchmark_runs", 0)
        if benchmark_runs > 0:
            _run_native_q8_vup_benchmark(
                row,
                args,
                kernel,
                q_weight,
                scales,
                biases,
                H,
                latent_dim,
            )
    except Exception as exc:
        row["native_q8_vup_smoke_error"] = repr(exc)


def _native_q8_vup_reference(x, q_weight, scales, biases):
    B, H, L, V = (*x.shape[:3], q_weight.shape[1])
    reference = mx.quantized_matmul(
        x,
        q_weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=64,
        bits=8,
        mode="affine",
    )
    return reference.transpose(0, 2, 1, 3).reshape(B, L, H * V)


def _native_indexer_smoke_reference(q, k, weights, *, causal=True):
    scores = q @ k.swapaxes(-1, -2)
    scores = mx.maximum(scores, 0)
    scores = scores * weights.swapaxes(-1, -2)[..., None]
    scores = scores.sum(axis=1, keepdims=True)
    if causal:
        L = q.shape[2]
        K = k.shape[2]
        q_positions = mx.arange(L).reshape(1, 1, L, 1)
        k_positions = mx.arange(K).reshape(1, 1, 1, K)
        causal_mask = k_positions <= (K - L + q_positions)
        scores = mx.where(causal_mask, scores, mx.array(-1e9, dtype=scores.dtype))
    return scores


def _run_native_indexer_smoke(row, args, status):
    if not status["available"]:
        row["native_indexer_smoke_error"] = "native DSA indexer kernels unavailable"
        return
    try:
        B = 1
        H = 32
        L = max(64, ((args.native_smoke_q_len + 63) // 64) * 64)
        K = max(4096, args.native_smoke_k_len)
        D = 128
        topk = 2048
        mx.random.seed(args.native_smoke_seed + 3)
        q = mx.random.normal((B, H, L, D), dtype=mx.float16) * 0.02
        k = mx.random.normal((B, 1, K, D), dtype=mx.float16) * 0.02
        weights = mx.random.normal((B, L, H), dtype=mx.float16) * 0.02
        scores = glm_moe_dsa._native_indexer_scores(
            q,
            k,
            weights,
            causal=True,
            skip_causal_future_store=False,
            causal_q_offset=K - L,
        )
        if scores is None:
            raise RuntimeError("native DSA indexer score kernel unavailable")
        reference = _native_indexer_smoke_reference(q, k, weights, causal=True)
        q_positions = mx.arange(L).reshape(1, 1, L, 1)
        k_positions = mx.arange(K).reshape(1, 1, 1, K)
        causal_mask = k_positions <= (K - L + q_positions)
        diff = mx.where(
            causal_mask,
            mx.abs(scores.astype(mx.float32) - reference.astype(mx.float32)),
            mx.zeros(scores.shape, dtype=mx.float32),
        )
        topk_indices = glm_moe_dsa._native_indexer_topk_indices(
            scores,
            topk,
            bucketed=True,
            causal_valid_prefix=True,
        )
        if topk_indices is None:
            raise RuntimeError("native DSA indexer top-k kernel unavailable")
        selected = mx.take_along_axis(
            reference,
            topk_indices.astype(mx.int32),
            axis=-1,
        )
        threshold = mx.sort(reference, axis=-1)[..., -topk:]
        threshold_min = mx.min(threshold, axis=-1, keepdims=True)
        topk_margin = mx.min(selected - threshold_min)
        mx.eval(scores, reference, diff, topk_indices, selected, threshold, topk_margin)

        max_abs_diff = float(mx.max(diff).item())
        mean_abs_diff = float(mx.mean(diff).item())
        min_topk_margin = float(topk_margin.item())
        row.update(
            {
                "native_indexer_smoke_available": status["available"],
                "native_indexer_smoke_source": status["source"],
                "native_indexer_smoke_import_error": status["import_error"],
                "native_indexer_smoke_score_shape": list(scores.shape),
                "native_indexer_smoke_topk_shape": list(topk_indices.shape),
                "native_indexer_smoke_max_abs_diff": max_abs_diff,
                "native_indexer_smoke_mean_abs_diff": mean_abs_diff,
                "native_indexer_smoke_min_topk_margin": min_topk_margin,
                "native_indexer_smoke_passed": (
                    max_abs_diff <= args.native_smoke_max_diff
                    and min_topk_margin >= -args.native_smoke_max_diff
                ),
            }
        )
    except Exception as exc:
        row["native_indexer_smoke_error"] = repr(exc)


def _run_native_q8_vup_benchmark(
    row,
    args,
    kernel,
    q_weight,
    scales,
    biases,
    heads,
    latent_dim,
):
    benchmark_runs = args.native_smoke_benchmark_runs
    benchmark_q_len = args.native_q8_vup_benchmark_q_len
    warmup_runs = args.native_smoke_benchmark_warmup_runs
    row.update(
        {
            "native_q8_vup_benchmark_runs": benchmark_runs,
            "native_q8_vup_benchmark_warmup_runs": warmup_runs,
            "native_q8_vup_benchmark_q_len": benchmark_q_len,
        }
    )
    try:
        mx.random.seed(args.native_smoke_seed + 2)
        x = mx.random.normal(
            (1, heads, benchmark_q_len, latent_dim), dtype=mx.float16
        ) * 0.02

        native_timings = benchmark_mx_callable(
            lambda: kernel(x, q_weight, scales, biases),
            benchmark_runs,
            warmup_runs,
        )
        reference_timings = benchmark_mx_callable(
            lambda: _native_q8_vup_reference(x, q_weight, scales, biases),
            benchmark_runs,
            warmup_runs,
        )
        native_mean = native_timings["mean"]
        reference_mean = reference_timings["mean"]
        row.update(
            {
                "native_q8_vup_native_seconds_mean": native_mean,
                "native_q8_vup_native_seconds_min": native_timings["min"],
                "native_q8_vup_native_seconds_p50": native_timings["p50"],
                "native_q8_vup_reference_seconds_mean": reference_mean,
                "native_q8_vup_reference_seconds_min": reference_timings["min"],
                "native_q8_vup_reference_seconds_p50": reference_timings["p50"],
                "native_q8_vup_speedup_mean": (
                    reference_mean / native_mean if native_mean > 0 else None
                ),
                "native_q8_vup_benchmark_error": None,
            }
        )
    except Exception as exc:
        row["native_q8_vup_benchmark_error"] = repr(exc)


def run_native_kernel_smoke(args):
    status = glm_moe_dsa.get_glm_dsa_native_sparse_prefill_status()
    indexer_status = glm_moe_dsa.get_glm_dsa_native_indexer_status()
    q8_vup_status = glm_moe_dsa.get_glm_dsa_native_q8_vup_status()
    row = {
        "case": "native-smoke",
        "mode": "native-smoke",
        "native_smoke_available": status["available"],
        "native_smoke_source": status["source"],
        "native_smoke_import_error": status["import_error"],
        "native_smoke_symbols": (),
        "native_smoke_q_len": args.native_smoke_q_len,
        "native_smoke_k_len": args.native_smoke_k_len,
        "native_smoke_seed": args.native_smoke_seed,
        "native_smoke_max_diff": args.native_smoke_max_diff,
        "native_smoke_shape": None,
        "native_smoke_max_abs_diff": None,
        "native_smoke_mean_abs_diff": None,
        "native_smoke_passed": False,
        "native_smoke_error": None,
        "native_indexer_smoke_available": indexer_status["available"],
        "native_indexer_smoke_source": indexer_status["source"],
        "native_indexer_smoke_import_error": indexer_status["import_error"],
        "native_indexer_smoke_score_shape": None,
        "native_indexer_smoke_topk_shape": None,
        "native_indexer_smoke_max_abs_diff": None,
        "native_indexer_smoke_mean_abs_diff": None,
        "native_indexer_smoke_min_topk_margin": None,
        "native_indexer_smoke_passed": False,
        "native_indexer_smoke_error": None,
        "native_q8_vup_smoke_available": q8_vup_status["available"],
        "native_q8_vup_smoke_source": q8_vup_status["source"],
        "native_q8_vup_smoke_import_error": q8_vup_status["import_error"],
        "native_q8_vup_smoke_shape": None,
        "native_q8_vup_smoke_max_abs_diff": None,
        "native_q8_vup_smoke_mean_abs_diff": None,
        "native_q8_vup_smoke_passed": False,
        "native_q8_vup_smoke_error": None,
        "native_q8_vup_benchmark_runs": getattr(
            args, "native_smoke_benchmark_runs", 0
        ),
        "native_q8_vup_benchmark_warmup_runs": getattr(
            args, "native_smoke_benchmark_warmup_runs", 0
        ),
        "native_q8_vup_benchmark_q_len": getattr(
            args, "native_q8_vup_benchmark_q_len", None
        ),
        "native_q8_vup_native_seconds_mean": None,
        "native_q8_vup_native_seconds_min": None,
        "native_q8_vup_native_seconds_p50": None,
        "native_q8_vup_reference_seconds_mean": None,
        "native_q8_vup_reference_seconds_min": None,
        "native_q8_vup_reference_seconds_p50": None,
        "native_q8_vup_speedup_mean": None,
        "native_q8_vup_benchmark_error": None,
    }
    row.update(_native_smoke_status_fields(args, status))
    if not status["available"]:
        row["native_smoke_error"] = "native sparse MLA kernel unavailable"
    else:
        try:
            symbols = _native_smoke_symbols(status["source"])
            kernel = _native_smoke_kernel(status["source"])
            if kernel is None:
                raise RuntimeError("native sparse MLA kernel unavailable")

            B, H = 1, 64
            L = args.native_smoke_q_len
            K = args.native_smoke_k_len
            latent_dim = 512
            rope_dim = 64
            mx.random.seed(args.native_smoke_seed)
            q_latent = mx.random.normal(
                (B, H, L, latent_dim), dtype=mx.float16
            ) * 0.02
            q_pe = mx.random.normal((B, H, L, rope_dim), dtype=mx.float16) * 0.02
            kv_latent = mx.random.normal(
                (B, 1, K, latent_dim), dtype=mx.float16
            ) * 0.02
            k_pe = mx.random.normal((B, 1, K, rope_dim), dtype=mx.float16) * 0.02
            topk_indices = mx.broadcast_to(
                mx.arange(K, dtype=mx.uint32).reshape(1, 1, 1, K),
                (B, 1, L, K),
            )
            scale = 1.0 / math.sqrt(latent_dim + rope_dim)

            output = kernel(
                q_latent,
                q_pe,
                kv_latent,
                k_pe,
                topk_indices,
                scale,
                causal=True,
            )
            reference = _native_smoke_dense_reference(
                q_latent, q_pe, kv_latent, k_pe, scale
            )
            diff = mx.abs(output.astype(mx.float32) - reference.astype(mx.float32))
            mx.eval(output, reference, diff)

            max_abs_diff = float(mx.max(diff).item())
            mean_abs_diff = float(mx.mean(diff).item())
            row.update(
                {
                    "native_smoke_symbols": symbols,
                    "native_smoke_shape": list(output.shape),
                    "native_smoke_max_abs_diff": max_abs_diff,
                    "native_smoke_mean_abs_diff": mean_abs_diff,
                    "native_smoke_passed": max_abs_diff <= args.native_smoke_max_diff,
                }
            )
        except Exception as exc:
            row["native_smoke_error"] = repr(exc)
    _run_native_indexer_smoke(row, args, indexer_status)
    _run_native_q8_vup_smoke(row, args, q8_vup_status)
    return row


def prefill_config_summary(args):
    return {
        "kv_bits": args.kv_bits,
        "kv_group_size": args.kv_group_size,
        "quantized_kv_start": args.quantized_kv_start,
        "prefill_step_size": args.prefill_step_size,
        "prefill_max_qk_tokens": getattr(
            args,
            "prefill_max_qk_tokens",
            DEFAULT_PREFILL_MAX_QK_TOKENS,
        ),
        "glm_dsa_adaptive_prefill_step_size": (
            getattr(args, "glm_dsa_adaptive_prefill_step_size", 0)
        ),
        "glm_dsa_adaptive_prefill_after_tokens": (
            getattr(args, "glm_dsa_adaptive_prefill_after_tokens", 0)
        ),
        "glm_dsa_adaptive_prefill_min_remaining_tokens": (
            getattr(args, "glm_dsa_adaptive_prefill_min_remaining_tokens", 0)
        ),
    }


def run_once(model, tokenizer, prompt, args, case_name):
    tokenize_t0 = time.perf_counter()
    tokens = prompt_to_tokens(tokenizer, prompt)
    if args.target_tokens is not None:
        tokens = tokens[: args.target_tokens]
    tokenize_seconds = time.perf_counter() - tokenize_t0

    progress_events = []
    prefill_stop_after_tokens = getattr(args, "prefill_stop_after_tokens", None)
    capture = CheckpointLogCapture()
    logger = logging.getLogger("mlx_lm.generate")
    old_level = logger.level
    old_debug = os.environ.get(PROMPT_CHECKPOINT_DEBUG_ENV)
    logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    os.environ[PROMPT_CHECKPOINT_DEBUG_ENV] = "1"

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    mx.clear_cache()
    mx.synchronize()
    reset_glm_dsa_profile()
    prefill_config = prefill_config_summary(args)

    def on_prompt_progress(done, total):
        progress_events.append((done, total, time.perf_counter()))
        if (
            prefill_stop_after_tokens is not None
            and done >= prefill_stop_after_tokens
            and done < total
        ):
            raise PrefillBenchmarkStop(done, total)

    ttft_t0 = time.perf_counter()
    response = None
    prefill_stop = None
    try:
        try:
            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=tokens,
                max_tokens=args.max_tokens,
                prefill_step_size=args.prefill_step_size,
                prefill_max_qk_tokens=prefill_config["prefill_max_qk_tokens"],
                glm_dsa_adaptive_prefill_step_size=(
                    prefill_config["glm_dsa_adaptive_prefill_step_size"]
                ),
                glm_dsa_adaptive_prefill_after_tokens=(
                    prefill_config["glm_dsa_adaptive_prefill_after_tokens"]
                ),
                glm_dsa_adaptive_prefill_min_remaining_tokens=(
                    prefill_config[
                        "glm_dsa_adaptive_prefill_min_remaining_tokens"
                    ]
                ),
                kv_bits=args.kv_bits,
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
                prompt_checkpoint=not args.no_prompt_checkpoint,
                prompt_checkpoint_store_prefix_lengths=(
                    args.checkpoint_store_prefix_lengths
                ),
                prompt_checkpoint_save_exact=(
                    args.checkpoint_save_exact == "enabled"
                ),
                prompt_checkpoint_frontier_min_tokens=(
                    args.checkpoint_frontier_min_tokens
                ),
                prompt_checkpoint_frontier_stride_tokens=(
                    args.checkpoint_frontier_stride_tokens
                ),
                prompt_progress_callback=on_prompt_progress,
            ):
                break
            mx.synchronize()
        except PrefillBenchmarkStop as exc:
            prefill_stop = exc
            mx.synchronize()
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
        if old_debug is None:
            os.environ.pop(PROMPT_CHECKPOINT_DEBUG_ENV, None)
        else:
            os.environ[PROMPT_CHECKPOINT_DEBUG_ENV] = old_debug

    ttft_seconds = time.perf_counter() - ttft_t0
    if response is None and prefill_stop is None:
        raise RuntimeError(f"{case_name}: generation produced no response")

    checkpoint = extract_checkpoint_summary(capture.messages)
    glm_profile = collect_glm_dsa_profile(args)
    prompt_tps = (
        response.prompt_tps
        if response is not None
        else prefill_stop.processed_tokens / ttft_seconds
    )
    peak_memory_gb = (
        response.peak_memory
        if response is not None
        else mx.get_peak_memory() / 1e9
    )
    finish_reason = (
        response.finish_reason
        if response is not None
        else "prefill-stop-after-tokens"
    )
    return {
        "case": case_name,
        "mode": "single",
        "batch_size": 1,
        "requested_total_tokens": args.target_tokens,
        "stored_prefix_tokens": None,
        "expected_reused_prefix_tokens": None,
        "checkpoint_expected_match": None,
        "checkpoint_cache_dir": args.resolved_checkpoint_cache_dir,
        "checkpoint_save_exact": args.checkpoint_save_exact,
        "prompt_tokens": len(tokens),
        "tokenize_seconds": tokenize_seconds,
        "ttft_seconds": ttft_seconds,
        "prompt_tps": prompt_tps,
        "peak_memory_gb": peak_memory_gb,
        "progress_events": len(progress_events),
        "finish_reason": finish_reason,
        "prefill_stopped_early": prefill_stop is not None,
        "prefill_stop_after_tokens": prefill_stop_after_tokens,
        "partial_prefill_tokens": (
            prefill_stop.processed_tokens if prefill_stop is not None else None
        ),
        "partial_prefill_total_tokens": (
            prefill_stop.total_tokens if prefill_stop is not None else None
        ),
        **prefill_config,
        **checkpoint,
        **glm_profile,
    }


def run_batch_once(model, tokenizer, text, args, case_name):
    tokenize_t0 = time.perf_counter()
    prompts = []
    for i in range(args.batch_size):
        prompt_text = f"{text}\n\nBatch request variant: {i}\n"
        tokens = encode(tokenizer, prompt_text)
        if args.target_tokens is not None:
            tokens = tokens[: args.target_tokens]
        prompts.append(tokens)
    tokenize_seconds = time.perf_counter() - tokenize_t0

    prefill_batch_size = args.prefill_batch_size or args.batch_size
    completion_batch_size = args.completion_batch_size or args.batch_size
    generator = BatchGenerator(
        model,
        max_tokens=args.max_tokens,
        prefill_batch_size=prefill_batch_size,
        completion_batch_size=completion_batch_size,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )
    uids = set(generator.insert(prompts))

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    mx.clear_cache()
    mx.synchronize()
    reset_glm_dsa_profile()

    stats = BatchStats()
    responses = {}
    completed = set()
    ttft_t0 = time.perf_counter()
    first_response_seconds = None
    active_batch_size_max = 0
    with generator.stats(stats):
        while len(completed) < len(uids):
            batch_responses = generator.next_generated()
            active_batch_size_max = max(
                active_batch_size_max,
                generator.admission_stats["active_batch_size"],
            )
            if batch_responses and first_response_seconds is None:
                mx.synchronize()
                first_response_seconds = time.perf_counter() - ttft_t0
            for response in batch_responses:
                responses[response.uid] = response
                if response.finish_reason is not None:
                    completed.add(response.uid)
            if not batch_responses:
                break
    mx.synchronize()
    admission_stats = stats.admission_stats
    generator.close()

    if first_response_seconds is None:
        raise RuntimeError(f"{case_name}: batched generation produced no response")

    return {
        "case": case_name,
        "mode": "batch",
        "batch_size": args.batch_size,
        "prefill_batch_size": prefill_batch_size,
        "completion_batch_size": completion_batch_size,
        "requested_total_tokens": args.target_tokens,
        "stored_prefix_tokens": None,
        "expected_reused_prefix_tokens": None,
        "checkpoint_expected_match": None,
        "checkpoint_cache_dir": args.resolved_checkpoint_cache_dir,
        "checkpoint_save_exact": args.checkpoint_save_exact,
        "prompt_tokens": sum(len(p) for p in prompts),
        "prompt_tokens_per_request": [len(p) for p in prompts],
        "tokenize_seconds": tokenize_seconds,
        "ttft_seconds": first_response_seconds,
        "prompt_tps": stats.prompt_tps,
        "peak_memory_gb": stats.peak_memory,
        "progress_events": None,
        "finish_reason": ",".join(
            sorted({str(r.finish_reason) for r in responses.values()})
        ),
        **prefill_config_summary(args),
        "checkpoint_resolution": "batch-n/a",
        "checkpoint_total_prompt_tokens": None,
        "server_cached_tokens": None,
        "disk_cached_tokens": None,
        "fresh_prompt_tokens": None,
        "fresh_prefill_tokens": None,
        "checkpoint_prefill_step_size": None,
        "checkpoint_lookup_seconds": None,
        "checkpoint_files_scanned": None,
        "checkpoint_candidates_scanned": None,
        "checkpoint_matched_candidates": None,
        "checkpoint_manifest_entries": None,
        "checkpoint_manifest_bootstrap": None,
        "checkpoint_events": 0,
        "ttft_p50_seconds": first_response_seconds,
        "ttft_p95_seconds": first_response_seconds,
        "admission_wait_p50_seconds": percentile(
            [
                event["wait_seconds"]
                for event in stats.admission_events
                if event["event"] == "admitted"
            ],
            50,
        ),
        "admission_wait_p95_seconds": percentile(
            [
                event["wait_seconds"]
                for event in stats.admission_events
                if event["event"] == "admitted"
            ],
            95,
        ),
        "active_batch_size_max": active_batch_size_max,
        **admission_stats,
        **collect_glm_dsa_profile(args),
    }


def run_queued_once(model, tokenizer, _text, args, case_name):
    tokenize_t0 = time.perf_counter()
    prompt_texts = build_queued_prompt_texts(tokenizer, args)
    prompts = [encode(tokenizer, text) for text in prompt_texts]
    tokenize_seconds = time.perf_counter() - tokenize_t0

    prefill_batch_size = args.prefill_batch_size or args.batch_size
    completion_batch_size = args.completion_batch_size or args.batch_size
    generator = BatchGenerator(
        model,
        max_tokens=args.max_tokens,
        prefill_batch_size=prefill_batch_size,
        completion_batch_size=completion_batch_size,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )

    insert_t0 = time.perf_counter()
    uids = set(generator.insert(prompts))
    insert_times = {uid: insert_t0 for uid in uids}

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    mx.clear_cache()
    mx.synchronize()
    reset_glm_dsa_profile()

    stats = BatchStats()
    responses = {}
    completed = set()
    ttft_by_uid = {}
    active_samples = []
    with generator.stats(stats):
        while len(completed) < len(uids):
            prompt_responses, generation_responses = generator.next()
            snapshot = generator.admission_stats
            active_samples.append(
                {
                    "elapsed_seconds": time.perf_counter() - insert_t0,
                    **snapshot,
                }
            )
            for response in generation_responses:
                responses[response.uid] = response
                ttft_by_uid.setdefault(
                    response.uid,
                    time.perf_counter() - insert_times[response.uid],
                )
                if response.finish_reason is not None:
                    completed.add(response.uid)
            if (
                not prompt_responses
                and not generation_responses
                and snapshot["queued_request_count"] == 0
                and snapshot["active_batch_size"] == 0
            ):
                break
    mx.synchronize()
    admission_stats = stats.admission_stats
    admission_events = stats.admission_events
    generator.close()

    if completed != uids:
        raise RuntimeError(
            f"{case_name}: queued generation completed {len(completed)} "
            f"of {len(uids)} requests"
        )

    wait_seconds = [
        event["wait_seconds"]
        for event in admission_events
        if event["event"] == "admitted"
    ]
    active_sizes = [sample["active_batch_size"] for sample in active_samples]
    result = {
        "case": case_name,
        "mode": "queued",
        "batch_size": args.batch_size,
        "prefill_batch_size": prefill_batch_size,
        "completion_batch_size": completion_batch_size,
        "requested_total_tokens": None,
        "stored_prefix_tokens": None,
        "expected_reused_prefix_tokens": None,
        "checkpoint_expected_match": None,
        "checkpoint_cache_dir": args.resolved_checkpoint_cache_dir,
        "checkpoint_save_exact": args.checkpoint_save_exact,
        "queued_requests": len(prompts),
        "prompt_tokens": sum(len(p) for p in prompts),
        "prompt_tokens_per_request": [len(p) for p in prompts],
        "tokenize_seconds": tokenize_seconds,
        "ttft_seconds": min(ttft_by_uid.values()),
        "ttft_p50_seconds": percentile(list(ttft_by_uid.values()), 50),
        "ttft_p95_seconds": percentile(list(ttft_by_uid.values()), 95),
        "prompt_tps": stats.prompt_tps,
        "peak_memory_gb": stats.peak_memory,
        "progress_events": None,
        "finish_reason": ",".join(
            sorted({str(r.finish_reason) for r in responses.values()})
        ),
        **prefill_config_summary(args),
        "checkpoint_resolution": "batch-n/a",
        "checkpoint_total_prompt_tokens": None,
        "server_cached_tokens": None,
        "disk_cached_tokens": None,
        "fresh_prompt_tokens": None,
        "fresh_prefill_tokens": None,
        "checkpoint_prefill_step_size": None,
        "checkpoint_lookup_seconds": None,
        "checkpoint_files_scanned": None,
        "checkpoint_candidates_scanned": None,
        "checkpoint_matched_candidates": None,
        "checkpoint_manifest_entries": None,
        "checkpoint_manifest_bootstrap": None,
        "checkpoint_events": 0,
        "admission_wait_p50_seconds": percentile(wait_seconds, 50),
        "admission_wait_p95_seconds": percentile(wait_seconds, 95),
        "active_batch_size_max": max(active_sizes) if active_sizes else 0,
        **admission_stats,
        **collect_glm_dsa_profile(args),
    }
    result["admission_events"] = admission_events
    result["active_batch_size_samples"] = active_samples
    return result


def summarize_repeats(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    summaries = []
    for case, case_rows in grouped.items():
        first = dict(case_rows[-1])
        if len(case_rows) > 1:
            first["ttft_seconds_median"] = statistics.median(
                r["ttft_seconds"] for r in case_rows
            )
            first["prompt_tps_median"] = statistics.median(
                r["prompt_tps"] for r in case_rows
            )
        summaries.append(first)
    return summaries


def format_output_cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def print_table(rows, output_format):
    headers = [
        "case",
        "mode",
        "policy_name",
        "policy_candidate_index",
        "policy_store_prefix_tokens",
        "policy_reused_ratio",
        "policy_fresh_prefill_ratio",
        "prefill_sweep_name",
        "prefill_sweep_candidate_index",
        "prefill_sweep_use_checkpoints",
        "prefill_sweep_fast_prefill_min_context",
        "from_q_a_kernel_sweep_name",
        "from_q_a_kernel_sweep_candidate_index",
        "batch_size",
        "prefill_batch_size",
        "completion_batch_size",
        "requested_total_tokens",
        "stored_prefix_tokens",
        "expected_reused_prefix_tokens",
        "disk_cached_tokens",
        "fresh_prompt_tokens",
        "fresh_prefill_tokens",
        "checkpoint_expected_match",
        "checkpoint_total_prompt_tokens",
        "server_cached_tokens",
        "checkpoint_prefill_step_size",
        "checkpoint_prefill_max_qk_tokens",
        "checkpoint_glm_dsa_adaptive_prefill_step_size",
        "checkpoint_glm_dsa_adaptive_prefill_after_tokens",
        "checkpoint_glm_dsa_adaptive_prefill_min_remaining_tokens",
        "checkpoint_prefill_chunks",
        "checkpoint_max_adaptive_prefill_step_size",
        "checkpoint_max_effective_prefill_step_size",
        "checkpoint_prefill_chunk_seconds_total",
        "checkpoint_max_prefill_chunk_seconds",
        "checkpoint_slowest_prefill_chunk_start_tokens",
        "checkpoint_slowest_prefill_chunk_tokens",
        "checkpoint_slowest_prefill_chunk_route",
        "checkpoint_native_sparse_prefill_chunks",
        "checkpoint_fast_sparse_prefill_chunks",
        "checkpoint_dense_prefill_chunks",
        "checkpoint_native_indexer_chunks",
        "checkpoint_cache_dir",
        "checkpoint_save_exact",
        "checkpoint_lookup_seconds",
        "checkpoint_files_scanned",
        "checkpoint_candidates_scanned",
        "checkpoint_matched_candidates",
        "checkpoint_manifest_entries",
        "checkpoint_manifest_bootstrap",
        "prefill_stopped_early",
        "prefill_stop_after_tokens",
        "partial_prefill_tokens",
        "partial_prefill_total_tokens",
        "prompt_tokens",
        "tokenize_seconds",
        "ttft_seconds",
        "ttft_p50_seconds",
        "ttft_p95_seconds",
        "prompt_tps",
        "kv_bits",
        "kv_group_size",
        "quantized_kv_start",
        "prefill_step_size",
        "prefill_max_qk_tokens",
        "glm_dsa_adaptive_prefill_step_size",
        "glm_dsa_adaptive_prefill_after_tokens",
        "glm_dsa_adaptive_prefill_min_remaining_tokens",
        "peak_memory_gb",
        "admission_wait_p50_seconds",
        "admission_wait_p95_seconds",
        "glm_mla_quantized_batch_admitted",
        "glm_mla_quantized_batch_rejected_mixed_cache",
        "glm_mla_waited_for_compatible_batch",
        "active_batch_cache_kind",
        "active_batch_size_max",
        "queued_request_count",
        "checkpoint_resolution",
        "glm_dsa_prefill_profile",
        "glm_dsa_prefill_profile_isolate",
        "glm_dsa_prefill_profile_isolate_env",
        "glm_dsa_fast_prefill",
        "glm_dsa_fast_prefill_env",
        "glm_dsa_fast_prefill_query_chunk",
        "glm_dsa_fast_prefill_key_block",
        "glm_dsa_sparse_prefill_min_context",
        "glm_dsa_native_sparse_prefill",
        "glm_dsa_native_sparse_prefill_env",
        "glm_dsa_native_sparse_prefill_available",
        "glm_dsa_native_sparse_prefill_source",
        "glm_dsa_native_sparse_prefill_import_error",
        "glm_dsa_native_sparse_prefill_min_context",
        "glm_dsa_native_sparse_prefill_quantized_kv",
        "glm_dsa_native_sparse_prefill_quantized_kv_env",
        "glm_dsa_native_sparse_prefill_quantized_kv_max_context",
        "glm_dsa_sparse_mla_tile_env",
        "glm_dsa_sparse_mla_tile",
        "glm_dsa_fast_prefill_hits",
        "glm_dsa_fast_prefill_fallback_reasons",
        "glm_dsa_native_sparse_prefill_hits",
        "glm_dsa_native_sparse_prefill_fallback_reasons",
        "glm_dsa_native_sparse_prefill_route_state",
        "glm_dsa_native_sparse_prefill_primary_fallback",
        "glm_dsa_native_sparse_prefill_config_blocker",
        "glm_dsa_native_sparse_prefill_attempt_min_context",
        "glm_dsa_native_indexer",
        "glm_dsa_native_indexer_env",
        "glm_dsa_native_indexer_available",
        "glm_dsa_native_indexer_source",
        "glm_dsa_native_indexer_import_error",
        "glm_dsa_native_indexer_scores_available",
        "glm_dsa_native_indexer_topk_available",
        "glm_dsa_native_indexer_min_context",
        "glm_dsa_native_indexer_hits",
        "glm_dsa_native_indexer_fallback_reasons",
        "glm_dsa_native_q8_vup",
        "glm_dsa_native_q8_vup_env",
        "glm_dsa_native_q8_vup_available",
        "glm_dsa_native_q8_vup_source",
        "glm_dsa_native_q8_vup_import_error",
        "glm_dsa_native_q8_vup_hits",
        "glm_dsa_native_q8_vup_fallback_reasons",
        "glm_dsa_native_q4_vup",
        "glm_dsa_native_q4_vup_env",
        "glm_dsa_native_q4_vup_available",
        "glm_dsa_native_q4_vup_source",
        "glm_dsa_native_q4_vup_import_error",
        "glm_dsa_native_q4_vup_hits",
        "glm_dsa_native_q4_vup_fallback_reasons",
        "glm_dsa_q_a_dense_cache",
        "glm_dsa_q_a_dense_cache_env",
        "glm_dsa_q_a_dense_cache_hits",
        "glm_dsa_q_a_dense_cache_builds",
        "glm_dsa_q_a_dense_cache_fallback_reasons",
        "glm_dsa_native_q_a_rms_norm",
        "glm_dsa_native_q_a_rms_norm_env",
        "glm_dsa_native_q_a_rms_norm_available",
        "glm_dsa_native_q_a_rms_norm_source",
        "glm_dsa_native_q_a_rms_norm_import_error",
        "glm_dsa_native_q_a_rms_norm_hits",
        "glm_dsa_native_q_a_rms_norm_fallback_reasons",
        "glm_dsa_native_q4_qa",
        "glm_dsa_native_q4_qa_env",
        "glm_dsa_native_q4_qa_tile_env",
        "glm_dsa_native_q4_qa_tile",
        "glm_dsa_native_q4_qa_available",
        "glm_dsa_native_q4_qa_source",
        "glm_dsa_native_q4_qa_import_error",
        "glm_dsa_native_q4_qa_hits",
        "glm_dsa_native_q4_qa_fallback_reasons",
        "glm_dsa_native_q4_qb",
        "glm_dsa_native_q4_qb_env",
        "glm_dsa_native_q4_qb_tile_env",
        "glm_dsa_native_q4_qb_tile",
        "glm_dsa_native_q4_qb_scaled_tile_env",
        "glm_dsa_native_q4_qb_scaled_tile",
        "glm_dsa_native_q4_qb_head_layout",
        "glm_dsa_native_q4_qb_head_layout_env",
        "glm_dsa_native_q4_qb_head_layout_available",
        "glm_dsa_native_q4_qb_head_layout_source",
        "glm_dsa_native_q4_qb_head_layout_import_error",
        "glm_dsa_native_q4_qb_from_q_a",
        "glm_dsa_native_q4_qb_from_q_a_env",
        "glm_dsa_native_q4_qb_from_q_a_kernel",
        "glm_dsa_native_q4_qb_from_q_a_kernel_env",
        "glm_dsa_native_q4_qb_from_q_a_available",
        "glm_dsa_native_q4_qb_from_q_a_rms_scale_source",
        "glm_dsa_native_q4_qb_from_q_a_projection_source",
        "glm_dsa_native_q4_qb_from_q_a_wscaled_heads_source",
        "glm_dsa_native_q4_qb_from_q_a_scaled_heads_source",
        "glm_dsa_native_q4_qb_from_q_a_import_error",
        "glm_dsa_native_q4_qb_available",
        "glm_dsa_native_q4_qb_source",
        "glm_dsa_native_q4_qb_import_error",
        "glm_dsa_native_q4_qb_hits",
        "glm_dsa_native_q4_qb_fallback_reasons",
        "glm_dsa_native_q4_qb_from_q_a_hits",
        "glm_dsa_native_q4_qb_from_q_a_fallback_reasons",
        "native_smoke_available",
        "native_smoke_source",
        "native_smoke_import_error",
        "native_smoke_symbols",
        "native_smoke_q_len",
        "native_smoke_k_len",
        "native_smoke_seed",
        "native_smoke_max_diff",
        "native_smoke_shape",
        "native_smoke_max_abs_diff",
        "native_smoke_mean_abs_diff",
        "native_smoke_passed",
        "native_smoke_error",
        "native_indexer_smoke_available",
        "native_indexer_smoke_source",
        "native_indexer_smoke_import_error",
        "native_indexer_smoke_score_shape",
        "native_indexer_smoke_topk_shape",
        "native_indexer_smoke_max_abs_diff",
        "native_indexer_smoke_mean_abs_diff",
        "native_indexer_smoke_min_topk_margin",
        "native_indexer_smoke_passed",
        "native_indexer_smoke_error",
        "native_q8_vup_smoke_available",
        "native_q8_vup_smoke_source",
        "native_q8_vup_smoke_import_error",
        "native_q8_vup_smoke_shape",
        "native_q8_vup_smoke_max_abs_diff",
        "native_q8_vup_smoke_mean_abs_diff",
        "native_q8_vup_smoke_passed",
        "native_q8_vup_smoke_error",
        "native_q8_vup_benchmark_runs",
        "native_q8_vup_benchmark_warmup_runs",
        "native_q8_vup_benchmark_q_len",
        "native_q8_vup_native_seconds_mean",
        "native_q8_vup_native_seconds_min",
        "native_q8_vup_native_seconds_p50",
        "native_q8_vup_reference_seconds_mean",
        "native_q8_vup_reference_seconds_min",
        "native_q8_vup_reference_seconds_p50",
        "native_q8_vup_speedup_mean",
        "native_q8_vup_benchmark_error",
        "glm_dsa_q_projection_seconds",
        "glm_dsa_q_a_projection_seconds",
        "glm_dsa_q_a_dense_cache_dequantization_seconds",
        "glm_dsa_q_a_dense_projection_seconds",
        "glm_dsa_native_q4_qa_projection_seconds",
        "glm_dsa_q_a_layernorm_seconds",
        "glm_dsa_native_q_a_rms_norm_seconds",
        "glm_dsa_q_b_projection_seconds",
        "glm_dsa_native_q4_qb_projection_seconds",
        "glm_dsa_native_q4_qb_head_layout_projection_seconds",
        "glm_dsa_native_q_a_rms_scale_seconds",
        "glm_dsa_native_q4_qb_from_q_a_projection_seconds",
        "glm_dsa_kv_cache_update_seconds",
        "glm_dsa_dsa_indexer_topk_seconds",
        "glm_dsa_native_indexer_scores_seconds",
        "glm_dsa_native_indexer_topk_seconds",
        "glm_dsa_latent_kv_dequantization_seconds",
        "glm_dsa_latent_kv_projection_seconds",
        "glm_dsa_native_sparse_kv_dequantization_seconds",
        "glm_dsa_native_q8_vup_seconds",
        "glm_dsa_native_q4_vup_seconds",
        "glm_dsa_sparse_gather_seconds",
        "glm_dsa_attention_seconds",
        "glm_dsa_native_sparse_attention_seconds",
        "glm_dsa_total_prefill_seconds",
    ]
    delimiter = "," if output_format == "csv" else "\t"
    writer = csv.writer(sys.stdout, delimiter=delimiter, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow([format_output_cell(row.get(h)) for h in headers])


def write_json_output(path, rows, *, partial=False):
    path = Path(path)
    payload = {"runs": rows, "summary": summarize_repeats(rows)}
    if partial:
        payload["partial"] = True
        payload["completed_runs"] = len(rows)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def partial_json_output_path(path):
    path = Path(path)
    return path.with_name(f"{path.name}.partial")


def remove_partial_json_output(path):
    partial_path = partial_json_output_path(path)
    try:
        partial_path.unlink()
    except FileNotFoundError:
        pass


def write_partial_prefill_sweep_output(args, rows):
    json_output = getattr(args, "json_output", None)
    if json_output:
        write_json_output(partial_json_output_path(json_output), rows, partial=True)


def set_sparse_prefill_min_context_env(value, fallback_env_value=None):
    env_key = glm_moe_dsa.GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV
    if value is not None:
        os.environ[env_key] = str(value)
    elif fallback_env_value is None:
        os.environ.pop(env_key, None)
    else:
        os.environ[env_key] = fallback_env_value


def configure_checkpoint_cache_dir(args):
    if args.mode in ("controlled-lcp", "policy-sweep") and args.checkpoint_cache_dir is None:
        args.checkpoint_cache_dir = Path(
            tempfile.mkdtemp(prefix=f"glm52-{args.mode}-checkpoints-")
        )
    if (
        args.mode == "prefill-sweep"
        and args.prefill_sweep_use_checkpoints
        and args.checkpoint_cache_dir is None
    ):
        args.checkpoint_cache_dir = Path(
            tempfile.mkdtemp(prefix="glm52-prefill-sweep-checkpoints-")
        )

    old_cache_dir = os.environ.get(prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV)
    if args.checkpoint_cache_dir is None:
        args.resolved_checkpoint_cache_dir = None
        return old_cache_dir

    resolved = args.checkpoint_cache_dir.expanduser().resolve()
    os.environ[prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV] = str(resolved)
    args.resolved_checkpoint_cache_dir = str(resolved)
    return old_cache_dir


def restore_checkpoint_cache_dir(old_cache_dir):
    if old_cache_dir is None:
        os.environ.pop(prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV, None)
    else:
        os.environ[prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV] = old_cache_dir


def controlled_lcp_lengths(args):
    prefix_tokens = (
        args.lcp_prefix_tokens
        if args.lcp_prefix_tokens is not None
        else args.repeat_prefix_tokens
    )
    suffix_tokens = (
        args.lcp_suffix_tokens
        if args.lcp_suffix_tokens is not None
        else args.repeat_suffix_tokens
    )
    if prefix_tokens <= 0:
        raise ValueError("--lcp-prefix-tokens must be positive")
    if suffix_tokens <= 0:
        raise ValueError("--lcp-suffix-tokens must be positive")

    requested_total_tokens = prefix_tokens + suffix_tokens
    store_lengths = args.checkpoint_store_prefix_lengths or [prefix_tokens]
    valid_store_lengths = sorted(
        {
            int(length)
            for length in store_lengths
            if (
                0 < int(length) < requested_total_tokens
                and int(length) <= prefix_tokens
            )
        }
    )
    if not valid_store_lengths:
        raise ValueError(
            "controlled LCP requires a store prefix length within the shared prefix"
        )
    return prefix_tokens, suffix_tokens, requested_total_tokens, valid_store_lengths


def build_controlled_lcp_prompts(tokenizer, prefix_tokens, suffix_tokens):
    prefix = build_prompt_tokens(tokenizer, prefix_tokens)
    suffix_a = build_prompt_tokens(
        tokenizer,
        suffix_tokens,
        prefix_text="\n\nControlled LCP seed suffix A.\n",
    )
    suffix_b = build_prompt_tokens(
        tokenizer,
        suffix_tokens,
        prefix_text="\n\nControlled LCP measured suffix B.\n",
    )
    return prefix + suffix_a, prefix + suffix_b


def run_controlled_lcp(model, tokenizer, args):
    (
        prefix_tokens,
        suffix_tokens,
        requested_total_tokens,
        store_lengths,
    ) = controlled_lcp_lengths(args)
    stored_prefix_tokens = 0 if args.no_prompt_checkpoint else max(store_lengths)
    expected_reused_prefix_tokens = stored_prefix_tokens
    seed_prompt, hit_prompt = build_controlled_lcp_prompts(
        tokenizer,
        prefix_tokens,
        suffix_tokens,
    )

    old_target_tokens = getattr(args, "target_tokens", None)
    old_store_lengths = args.checkpoint_store_prefix_lengths
    old_frontier_min = args.checkpoint_frontier_min_tokens
    args.target_tokens = requested_total_tokens
    args.checkpoint_store_prefix_lengths = store_lengths
    args.checkpoint_frontier_min_tokens = max(
        args.checkpoint_frontier_min_tokens,
        requested_total_tokens + 1,
    )
    try:
        seed_row = run_once(
            model,
            tokenizer,
            seed_prompt,
            args,
            "controlled-lcp-store",
        )
        seed_row.update(
            {
                "requested_total_tokens": requested_total_tokens,
                "stored_prefix_tokens": stored_prefix_tokens,
                "expected_reused_prefix_tokens": 0,
                "checkpoint_expected_match": (
                    seed_row.get("disk_cached_tokens") == 0
                ),
                "lcp_prefix_tokens": prefix_tokens,
                "lcp_suffix_tokens": suffix_tokens,
            }
        )

        hit_row = run_once(
            model,
            tokenizer,
            hit_prompt,
            args,
            "controlled-lcp-hit",
        )
        hit_row.update(
            {
                "requested_total_tokens": requested_total_tokens,
                "stored_prefix_tokens": stored_prefix_tokens,
                "expected_reused_prefix_tokens": expected_reused_prefix_tokens,
                "checkpoint_expected_match": (
                    hit_row.get("disk_cached_tokens")
                    == expected_reused_prefix_tokens
                ),
                "lcp_prefix_tokens": prefix_tokens,
                "lcp_suffix_tokens": suffix_tokens,
            }
        )
        if not args.no_prompt_checkpoint and not hit_row["checkpoint_expected_match"]:
            raise RuntimeError(
                "controlled LCP checkpoint mismatch: "
                f"expected disk_cached_tokens={expected_reused_prefix_tokens}, "
                f"got {hit_row.get('disk_cached_tokens')} "
                f"(resolution={hit_row.get('checkpoint_resolution')})"
            )
        return [seed_row, hit_row]
    finally:
        args.target_tokens = old_target_tokens
        args.checkpoint_store_prefix_lengths = old_store_lengths
        args.checkpoint_frontier_min_tokens = old_frontier_min


def run_policy_sweep(model, tokenizer, args):
    prefix_tokens = (
        args.lcp_prefix_tokens
        if args.lcp_prefix_tokens is not None
        else args.repeat_prefix_tokens
    )
    suffix_tokens = (
        args.lcp_suffix_tokens
        if args.lcp_suffix_tokens is not None
        else args.repeat_suffix_tokens
    )
    if prefix_tokens <= 0:
        raise ValueError("--lcp-prefix-tokens or --repeat-prefix-tokens must be positive")
    if suffix_tokens <= 0:
        raise ValueError("--lcp-suffix-tokens or --repeat-suffix-tokens must be positive")

    requested_total_tokens = prefix_tokens + suffix_tokens
    seed_prompt, hit_prompt = build_controlled_lcp_prompts(
        tokenizer,
        prefix_tokens,
        suffix_tokens,
    )
    candidates = policy_sweep_candidates(args, prefix_tokens, requested_total_tokens)
    base_cache_dir = Path(args.resolved_checkpoint_cache_dir)

    rows = []
    old_target_tokens = getattr(args, "target_tokens", None)
    old_store_lengths = args.checkpoint_store_prefix_lengths
    old_frontier_min = args.checkpoint_frontier_min_tokens
    old_no_prompt_checkpoint = args.no_prompt_checkpoint
    old_cache_dir = os.environ.get(prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV)
    old_resolved_cache_dir = args.resolved_checkpoint_cache_dir
    args.target_tokens = requested_total_tokens
    args.checkpoint_frontier_min_tokens = max(
        args.checkpoint_frontier_min_tokens,
        requested_total_tokens + 1,
    )
    try:
        for policy_index, (policy_name, store_length) in enumerate(candidates):
            candidate_dir = base_cache_dir / f"{policy_index:02d}-{safe_case_name(policy_name)}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            os.environ[prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV] = str(candidate_dir)
            args.resolved_checkpoint_cache_dir = str(candidate_dir)
            args.no_prompt_checkpoint = store_length == 0
            args.checkpoint_store_prefix_lengths = (
                None if store_length == 0 else [store_length]
            )
            expected_reused_prefix_tokens = 0 if store_length == 0 else store_length

            seed_row = run_once(
                model,
                tokenizer,
                seed_prompt,
                args,
                f"policy-{policy_name}-store",
            )
            seed_row.update(
                {
                    "mode": "policy-sweep",
                    "policy_name": policy_name,
                    "policy_store_prefix_tokens": store_length,
                    "policy_candidate_index": policy_index,
                    "requested_total_tokens": requested_total_tokens,
                    "stored_prefix_tokens": expected_reused_prefix_tokens,
                    "expected_reused_prefix_tokens": 0,
                    "checkpoint_expected_match": (
                        seed_row.get("disk_cached_tokens") == 0
                    ),
                    "lcp_prefix_tokens": prefix_tokens,
                    "lcp_suffix_tokens": suffix_tokens,
                }
            )
            rows.append(seed_row)

            hit_row = run_once(
                model,
                tokenizer,
                hit_prompt,
                args,
                f"policy-{policy_name}-hit",
            )
            hit_row.update(
                {
                    "mode": "policy-sweep",
                    "policy_name": policy_name,
                    "policy_store_prefix_tokens": store_length,
                    "policy_candidate_index": policy_index,
                    "requested_total_tokens": requested_total_tokens,
                    "stored_prefix_tokens": expected_reused_prefix_tokens,
                    "expected_reused_prefix_tokens": expected_reused_prefix_tokens,
                    "checkpoint_expected_match": (
                        hit_row.get("disk_cached_tokens")
                        == expected_reused_prefix_tokens
                    ),
                    "policy_reused_ratio": (
                        expected_reused_prefix_tokens / requested_total_tokens
                    ),
                    "policy_fresh_prefill_ratio": (
                        (hit_row.get("fresh_prefill_tokens") or 0)
                        / max(1, requested_total_tokens - 1)
                    ),
                    "lcp_prefix_tokens": prefix_tokens,
                    "lcp_suffix_tokens": suffix_tokens,
                }
            )
            if not hit_row["checkpoint_expected_match"]:
                raise RuntimeError(
                    "policy sweep checkpoint mismatch: "
                    f"policy={policy_name} "
                    f"expected disk_cached_tokens={expected_reused_prefix_tokens}, "
                    f"got {hit_row.get('disk_cached_tokens')} "
                    f"(resolution={hit_row.get('checkpoint_resolution')})"
                )
            rows.append(hit_row)
        return rows
    finally:
        args.target_tokens = old_target_tokens
        args.checkpoint_store_prefix_lengths = old_store_lengths
        args.checkpoint_frontier_min_tokens = old_frontier_min
        args.no_prompt_checkpoint = old_no_prompt_checkpoint
        args.resolved_checkpoint_cache_dir = old_resolved_cache_dir
        if old_cache_dir is None:
            os.environ.pop(prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV, None)
        else:
            os.environ[prompt_cache.PROMPT_CHECKPOINT_CACHE_DIR_ENV] = old_cache_dir


def run_prefill_sweep(model, tokenizer, args):
    candidates = prefill_sweep_candidates(args)
    if args.prompt_file:
        prompt_cases = [("file", args.prompt_file.read_text(), None)]
    else:
        prompt_cases = [
            (str(length), build_prompt_text(tokenizer, length), length)
            for length in parse_lengths(args.lengths)
        ]

    rows = []
    old_target_tokens = getattr(args, "target_tokens", None)
    old_prefill_step_size = args.prefill_step_size
    old_prefill_max_qk_tokens = args.prefill_max_qk_tokens
    old_adaptive_step_size = args.glm_dsa_adaptive_prefill_step_size
    old_fast_prefill_min_context = getattr(args, "fast_prefill_min_context", None)
    old_sparse_prefill_min_context_env = os.environ.get(
        glm_moe_dsa.GLM_DSA_SPARSE_PREFILL_MIN_CONTEXT_ENV
    )
    old_no_prompt_checkpoint = args.no_prompt_checkpoint
    old_checkpoint_save_exact = args.checkpoint_save_exact
    args.no_prompt_checkpoint = not args.prefill_sweep_use_checkpoints
    if not args.prefill_sweep_use_checkpoints:
        args.checkpoint_save_exact = "disabled"

    try:
        for prompt_label, prompt_text, target_tokens in prompt_cases:
            args.target_tokens = target_tokens
            for candidate_index, candidate in enumerate(candidates):
                args.prefill_step_size = candidate["prefill_step_size"]
                args.prefill_max_qk_tokens = candidate["prefill_max_qk_tokens"]
                args.glm_dsa_adaptive_prefill_step_size = candidate[
                    "glm_dsa_adaptive_prefill_step_size"
                ]
                args.fast_prefill_min_context = candidate["fast_prefill_min_context"]
                set_sparse_prefill_min_context_env(
                    args.fast_prefill_min_context,
                    old_sparse_prefill_min_context_env,
                )
                case_name = f"prefill-sweep-{prompt_label}-{candidate['name']}"
                for _run in range(args.repeat_runs):
                    row = run_once(model, tokenizer, prompt_text, args, case_name)
                    row.update(
                        {
                            "mode": "prefill-sweep",
                            "prefill_sweep_name": candidate["name"],
                            "prefill_sweep_candidate_index": candidate_index,
                            "prefill_sweep_use_checkpoints": (
                                args.prefill_sweep_use_checkpoints
                            ),
                            "prefill_sweep_fast_prefill_min_context": (
                                args.fast_prefill_min_context
                            ),
                        }
                    )
                    rows.append(row)
                    write_partial_prefill_sweep_output(args, rows)
        return rows
    finally:
        args.target_tokens = old_target_tokens
        args.prefill_step_size = old_prefill_step_size
        args.prefill_max_qk_tokens = old_prefill_max_qk_tokens
        args.glm_dsa_adaptive_prefill_step_size = old_adaptive_step_size
        args.fast_prefill_min_context = old_fast_prefill_min_context
        set_sparse_prefill_min_context_env(
            old_fast_prefill_min_context,
            old_sparse_prefill_min_context_env,
        )
        args.no_prompt_checkpoint = old_no_prompt_checkpoint
        args.checkpoint_save_exact = old_checkpoint_save_exact


def run_with_from_q_a_kernel_sweep(
    runner,
    model,
    tokenizer,
    text,
    args,
    case_name,
):
    candidates = getattr(args, "native_q4_qb_from_q_a_kernel_sweep", None)
    if not candidates:
        return [runner(model, tokenizer, text, args, case_name)]

    rows = []
    old_from_q_a_arg = args.native_q4_qb_from_q_a
    old_kernel_arg = args.native_q4_qb_from_q_a_kernel
    old_from_q_a_env = os.environ.get(glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV)
    old_kernel_env = os.environ.get(
        glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV
    )
    try:
        for candidate_index, candidate in enumerate(candidates):
            if candidate == "disabled":
                args.native_q4_qb_from_q_a = "disabled"
                args.native_q4_qb_from_q_a_kernel = "default"
                os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV] = "0"
                if old_kernel_env is None:
                    os.environ.pop(
                        glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV,
                        None,
                    )
                else:
                    os.environ[
                        glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV
                    ] = old_kernel_env
            else:
                args.native_q4_qb_from_q_a = "enabled"
                args.native_q4_qb_from_q_a_kernel = candidate
                os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV] = "1"
                os.environ[
                    glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV
                ] = candidate

            row = runner(
                model,
                tokenizer,
                text,
                args,
                f"{case_name}-fromqa-{candidate}",
            )
            row.update(
                {
                    "from_q_a_kernel_sweep_name": candidate,
                    "from_q_a_kernel_sweep_candidate_index": candidate_index,
                }
            )
            rows.append(row)
        return rows
    finally:
        args.native_q4_qb_from_q_a = old_from_q_a_arg
        args.native_q4_qb_from_q_a_kernel = old_kernel_arg
        if old_from_q_a_env is None:
            os.environ.pop(glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV, None)
        else:
            os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_ENV] = (
                old_from_q_a_env
            )
        if old_kernel_env is None:
            os.environ.pop(
                glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV,
                None,
            )
        else:
            os.environ[glm_moe_dsa.GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL_ENV] = (
                old_kernel_env
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="",
        help="Local path or HF repo. Not required for --mode native-smoke.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "single",
            "batch",
            "queued",
            "controlled-lcp",
            "policy-sweep",
            "prefill-sweep",
            "native-smoke",
        ),
        default="single",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-batch-size", type=int)
    parser.add_argument("--completion-batch-size", type=int)
    parser.add_argument("--lengths", default="128,8192,32768")
    parser.add_argument("--queued-requests", type=int, default=8)
    parser.add_argument("--queued-prefix-tokens", type=int, default=2048)
    parser.add_argument("--queued-suffix-tokens", default="128,512,2048")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--repeat-prefix-tokens", type=int, default=8192)
    parser.add_argument("--repeat-suffix-tokens", type=int, default=256)
    parser.add_argument(
        "--lcp-prefix-tokens",
        type=int,
        help=(
            "Shared prefix tokens for --mode controlled-lcp. Defaults to "
            "--repeat-prefix-tokens."
        ),
    )
    parser.add_argument(
        "--lcp-suffix-tokens",
        type=int,
        help=(
            "Changed suffix tokens for --mode controlled-lcp. Defaults to "
            "--repeat-suffix-tokens."
        ),
    )
    parser.add_argument(
        "--policy-candidates",
        type=parse_policy_candidates,
        help=(
            "Comma-separated policy candidates for --mode policy-sweep. "
            "Use name:length entries; length 0 is the disabled baseline. "
            "When omitted, the sweep compares disabled, ds4-boundary, and "
            "full-prefix when distinct."
        ),
    )
    parser.add_argument(
        "--policy-min-tokens",
        type=int,
        default=512,
        help="Minimum prefix length used when deriving the default ds4-boundary candidate.",
    )
    parser.add_argument(
        "--policy-boundary-trim-tokens",
        type=int,
        default=32,
        help="Tail trim used when deriving the default ds4-boundary candidate.",
    )
    parser.add_argument(
        "--policy-boundary-align-tokens",
        type=int,
        default=2048,
        help="Alignment used when deriving the default ds4-boundary candidate.",
    )
    parser.add_argument(
        "--prefill-step-candidates",
        type=parse_lengths,
        help=(
            "Comma-separated base prefill step candidates for --mode prefill-sweep. "
            "Defaults to 512,1024,2048."
        ),
    )
    parser.add_argument(
        "--prefill-max-qk-token-candidates",
        type=parse_lengths,
        help=(
            "Comma-separated QK budget candidates for --mode prefill-sweep. "
            "Use 0 to disable context-aware shrinking. Defaults to "
            "--prefill-max-qk-tokens."
        ),
    )
    parser.add_argument(
        "--glm-dsa-adaptive-prefill-step-candidates",
        type=parse_lengths,
        help=(
            "Comma-separated adaptive GLM DSA step candidates for "
            "--mode prefill-sweep. Use 0 to disable. Defaults to 0,8192."
        ),
    )
    parser.add_argument(
        "--fast-prefill-min-context-candidates",
        type=parse_lengths,
        help=(
            "Comma-separated GLM DSA sparse prefill handoff thresholds for "
            "--mode prefill-sweep. When omitted, --fast-prefill-min-context "
            "or the current environment/default is used."
        ),
    )
    parser.add_argument(
        "--prefill-stop-after-tokens",
        type=int,
        help=(
            "Benchmark-only early stop: abort a single-request run after "
            "prefill reaches this many processed prompt tokens and record a "
            "partial row. Useful for long-context prefill-sweep probes."
        ),
    )
    parser.add_argument(
        "--prefill-sweep-use-checkpoints",
        action="store_true",
        help=(
            "Allow prompt checkpoint load/save during --mode prefill-sweep. "
            "By default prefill-sweep disables checkpoints to measure cold "
            "prefill step behavior."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument(
        "--prefill-max-qk-tokens",
        type=int,
        default=DEFAULT_PREFILL_MAX_QK_TOKENS,
        help=(
            "Maximum chunk_tokens * effective_context_tokens for single-request "
            "prefill. Use 0 to disable context-aware step shrinking."
        ),
    )
    parser.add_argument(
        "--glm-dsa-adaptive-prefill-step-size",
        type=int,
        default=0,
        help=(
            "Opt-in larger base prefill step for GLM DSA single-request runs "
            "before the context-aware QK cap is applied."
        ),
    )
    parser.add_argument(
        "--glm-dsa-adaptive-prefill-after-tokens",
        type=int,
        default=0,
        help="Minimum processed prompt tokens before adaptive GLM DSA prefill activates.",
    )
    parser.add_argument(
        "--glm-dsa-adaptive-prefill-min-remaining-tokens",
        type=int,
        default=0,
        help="Minimum remaining prompt tokens required for adaptive GLM DSA prefill.",
    )
    parser.add_argument("--kv-bits", type=int)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument(
        "--checkpoint-store-prefix-lengths",
        type=parse_lengths,
        default=None,
        help=(
            "Comma-separated token prefix lengths to save as reusable prompt "
            "checkpoints during single-request runs."
        ),
    )
    parser.add_argument(
        "--checkpoint-frontier-min-tokens",
        type=int,
        default=8192,
        help="First automatic frontier checkpoint length for single-request runs.",
    )
    parser.add_argument(
        "--checkpoint-frontier-stride-tokens",
        type=int,
        default=16384,
        help="Stride for automatic frontier checkpoint lengths.",
    )
    parser.add_argument(
        "--checkpoint-save-exact",
        choices=("enabled", "disabled"),
        default=None,
        help=(
            "Save the final exact prompt checkpoint after prefill. Defaults to "
            "enabled except in --mode controlled-lcp."
        ),
    )
    parser.add_argument(
        "--no-save-exact-checkpoint",
        action="store_const",
        const="disabled",
        dest="checkpoint_save_exact",
        help="Alias for --checkpoint-save-exact disabled.",
    )
    parser.add_argument(
        "--checkpoint-cache-dir",
        type=Path,
        help=(
            "Directory for prompt checkpoint files and manifest for this run. "
            "In controlled-lcp mode, a private temporary directory is created "
            "when this is omitted."
        ),
    )
    parser.add_argument(
        "--fast-prefill",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control GLM DSA sparse prefill fast path. The default leaves "
            "MLX_LM_GLM_DSA_FAST_PREFILL unchanged; unset means enabled."
        ),
    )
    parser.add_argument(
        "--fast-prefill-query-chunk",
        type=int,
        help="Query microbatch size for the GLM DSA sparse prefill gather path.",
    )
    parser.add_argument(
        "--fast-prefill-key-block",
        type=int,
        help="Key block size for the GLM DSA sparse prefill indexer path.",
    )
    parser.add_argument(
        "--fast-prefill-min-context",
        type=int,
        help=(
            "Minimum effective context length before using the GLM DSA sparse "
            "prefill path."
        ),
    )
    parser.add_argument(
        "--native-sparse-prefill",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the optional GLM DSA native sparse MLA prefill route. "
            "The default leaves MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL unchanged; "
            "unset means enabled but only used when a compatible native symbol "
            "and shape are available."
        ),
    )
    parser.add_argument(
        "--native-sparse-prefill-min-context",
        type=int,
        help=(
            "Minimum effective context length before trying the native sparse "
            "MLA route."
        ),
    )
    parser.add_argument(
        "--native-sparse-quantized-kv",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in native sparse MLA route for int8 GLM MLA KV "
            "cache. It temporarily dequantizes the full latent KV cache for "
            "the native kernel; default leaves "
            "MLX_LM_GLM_DSA_NATIVE_SPARSE_PREFILL_QUANTIZED_KV unchanged, "
            "and unset means disabled."
        ),
    )
    parser.add_argument(
        "--native-sparse-quantized-kv-max-context",
        type=int,
        help=(
            "Maximum effective context length allowed for the opt-in native "
            "sparse MLA route over int8 GLM MLA KV cache. Use 0 for no limit."
        ),
    )
    parser.add_argument(
        "--native-sparse-mla-tile",
        choices=(
            "default",
            "bk128",
            "bk256",
            "bk128_dc64",
            "wm4",
            "bk128_wm4",
            "bk128_dc64_wm4",
        ),
        default="default",
        help=(
            "Select the experimental native sparse MLA tile. The default "
            "leaves MLX_LM_GLM_DSA_SPARSE_MLA_TILE unchanged; unset means "
            "bk256_dc32_wm8."
        ),
    )
    parser.add_argument(
        "--native-indexer",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the optional native GLM DSA indexer score/top-k route. "
            "The default leaves MLX_LM_GLM_DSA_NATIVE_INDEXER unchanged; "
            "unset means enabled when compatible vendored symbols and shapes "
            "are available."
        ),
    )
    parser.add_argument(
        "--native-q8-vup",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the optional native q8 V-up projection for quantized GLM "
            "DSA unembed_out weights. The default leaves "
            "MLX_LM_GLM_DSA_NATIVE_Q8_VUP unchanged; unset means disabled."
        ),
    )
    parser.add_argument(
        "--native-q4-vup",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in native q4 V-up projection for quantized GLM "
            "DSA unembed_out weights. The default leaves "
            "MLX_LM_GLM_DSA_NATIVE_Q4_VUP unchanged; unset means disabled."
        ),
    )
    parser.add_argument(
        "--native-q4-qb",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in native q4 q_b projection for GLM-5.2 M3 "
            "attention. The default leaves MLX_LM_GLM_DSA_NATIVE_Q4_QB "
            "unchanged; unset means disabled."
        ),
    )
    parser.add_argument(
        "--native-q4-qb-tile",
        choices=(
            "default",
            "bk32",
            "bk64",
            "bm16",
            "bn16",
            "bn64",
            "bm64",
            "bm16bn64",
            "bm64bn64",
            "bk64bn64",
        ),
        default="default",
        help=(
            "Select the opt-in native q4 q_b projection tile. The default "
            "leaves MLX_LM_GLM_DSA_NATIVE_Q4_QB_TILE unchanged; unset means "
            "bm64."
        ),
    )
    parser.add_argument(
        "--native-q4-qb-scaled-tile",
        choices=(
            "default",
            "bk32",
            "bk64",
            "bm16",
            "bn16",
            "bn64",
            "bm64",
            "bm16bn64",
            "bm64bn64",
            "bk64bn64",
        ),
        default="default",
        help=(
            "Select the q_b-from-q_a scaled native q4 q_b tile. The default "
            "leaves MLX_LM_GLM_DSA_NATIVE_Q4_QB_SCALED_TILE unchanged; unset "
            "means bn64."
        ),
    )
    parser.add_argument(
        "--native-q4-qb-head-layout",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in native q4 q_b projection variant that writes "
            "directly to [B,H,L,D] layout for q_projection experiments. This "
            "requires native q4 q_b to be enabled. The default leaves "
            "MLX_LM_GLM_DSA_NATIVE_Q4_QB_HEAD_LAYOUT unchanged; unset means "
            "disabled."
        ),
    )
    parser.add_argument(
        "--native-q4-qb-from-q-a",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in shared-layer q_projection variant that skips "
            "materializing qr and feeds q_a plus RMS scale directly into native "
            "q4 q_b. This requires native q4 q_b to be enabled. The default "
            "leaves MLX_LM_GLM_DSA_NATIVE_Q4_QB_FROM_Q_A unchanged; unset "
            "means disabled."
        ),
    )
    parser.add_argument(
        "--native-q4-qb-from-q-a-kernel",
        choices=("default", "scaled", "wscaled", "auto"),
        default="default",
        help=(
            "Select the q_b-from-q_a projection kernel. The default leaves "
            "MLX_LM_GLM_DSA_NATIVE_Q4_QB_FROM_Q_A_KERNEL unchanged; unset "
            "means scaled."
        ),
    )
    parser.add_argument(
        "--native-q4-qb-from-q-a-kernel-sweep",
        type=parse_from_q_a_kernel_sweep,
        help=(
            "Comma-separated single-mode sweep over q_b-from-q_a kernels. "
            "Candidates are disabled, scaled, wscaled, and auto. The disabled "
            "candidate measures the materialized-qr baseline."
        ),
    )
    parser.add_argument(
        "--q-a-dense-cache",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in dense q_a projection cache for GLM-5.2 M3 "
            "attention. It dequantizes q_a_proj weights once per layer and "
            "reuses the dense weight for later prefill calls. The default "
            "leaves MLX_LM_GLM_DSA_Q_A_DENSE_CACHE unchanged; unset means "
            "disabled."
        ),
    )
    parser.add_argument(
        "--native-q-a-rms-norm",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in native GLM q_a RMSNorm kernel for "
            "q_projection structure experiments. The default leaves "
            "MLX_LM_GLM_DSA_NATIVE_Q_A_RMS_NORM unchanged; unset means "
            "disabled."
        ),
    )
    parser.add_argument(
        "--native-q4-qa",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control the opt-in native q4 q_a projection for GLM-5.2 M3 "
            "attention. The default leaves MLX_LM_GLM_DSA_NATIVE_Q4_QA "
            "unchanged; unset means disabled."
        ),
    )
    parser.add_argument(
        "--native-q4-qa-tile",
        choices=(
            "default",
            "bk32",
            "bk64",
            "bm16",
            "bn16",
            "bn64",
            "bm64",
            "bm16bn64",
            "bm64bn64",
        ),
        default="default",
        help=(
            "Select the opt-in native q4 q_a projection tile. The default "
            "leaves MLX_LM_GLM_DSA_NATIVE_Q4_QA_TILE unchanged; unset means "
            "bk64."
        ),
    )
    parser.add_argument(
        "--native-smoke-q-len",
        type=int,
        default=2,
        help="Query length for --mode native-smoke. Must be greater than 1.",
    )
    parser.add_argument(
        "--native-smoke-k-len",
        type=int,
        default=32,
        help="Context length for --mode native-smoke. Must be at least 16.",
    )
    parser.add_argument(
        "--native-smoke-seed",
        type=int,
        default=7,
        help="Random seed for --mode native-smoke.",
    )
    parser.add_argument(
        "--native-smoke-max-diff",
        type=float,
        default=0.02,
        help="Maximum allowed absolute difference for --mode native-smoke.",
    )
    parser.add_argument(
        "--native-smoke-benchmark-runs",
        type=int,
        default=0,
        help=(
            "Optional timing runs for --mode native-smoke. When positive, "
            "benchmarks native q8 V-up against the MLX quantized_matmul reference."
        ),
    )
    parser.add_argument(
        "--native-smoke-benchmark-warmup-runs",
        type=int,
        default=2,
        help="Warmup runs before native-smoke q8 V-up timing.",
    )
    parser.add_argument(
        "--native-q8-vup-benchmark-q-len",
        type=int,
        default=256,
        help="Query length used for native-smoke q8 V-up timing.",
    )
    parser.add_argument(
        "--prefill-profile",
        action="store_true",
        help=(
            "Synchronize and report GLM DSA prefill stage timings. This adds "
            "profiling overhead and is intended for measurement runs."
        ),
    )
    parser.add_argument(
        "--prefill-profile-isolate",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "Control profiling-only input synchronization before selected GLM "
            "DSA stages. Enabling it reduces lazy-evaluation attribution drift "
            "in q_projection sub-stage timings."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-prompt-checkpoint", action="store_true")
    parser.add_argument(
        "--output-format",
        choices=("tsv", "csv"),
        default="tsv",
        help="Tab-separated or comma-separated console table output.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    args.model = (args.model or "").strip()
    if args.mode != "native-smoke" and not args.model:
        parser.error(
            "--model is empty; set MODEL to your model directory or pass an "
            "explicit --model /path/to/model value."
        )
    if (
        args.prefill_stop_after_tokens is not None
        and args.prefill_stop_after_tokens <= 0
    ):
        parser.error("--prefill-stop-after-tokens must be positive when set.")
    if args.native_q4_qb_from_q_a_kernel_sweep and args.mode != "single":
        parser.error("--native-q4-qb-from-q-a-kernel-sweep requires --mode single")
    if args.mode == "native-smoke":
        if args.native_smoke_q_len <= 1:
            parser.error("--native-smoke-q-len must be greater than 1.")
        if args.native_smoke_k_len < 16:
            parser.error("--native-smoke-k-len must be at least 16.")
        if args.native_smoke_k_len < args.native_smoke_q_len:
            parser.error("--native-smoke-k-len must be >= --native-smoke-q-len.")
        if args.native_smoke_max_diff < 0:
            parser.error("--native-smoke-max-diff must be non-negative.")
        if args.native_smoke_benchmark_runs < 0:
            parser.error("--native-smoke-benchmark-runs must be non-negative.")
        if args.native_smoke_benchmark_warmup_runs < 0:
            parser.error(
                "--native-smoke-benchmark-warmup-runs must be non-negative."
            )
        if args.native_q8_vup_benchmark_q_len <= 0:
            parser.error("--native-q8-vup-benchmark-q-len must be positive.")
    if args.checkpoint_save_exact is None:
        args.checkpoint_save_exact = (
            "disabled"
            if args.mode in ("controlled-lcp", "policy-sweep", "prefill-sweep")
            else "enabled"
        )
    args.target_tokens = None
    configure_glm_dsa_fast_prefill(args)
    if args.mode == "native-smoke":
        rows = [run_native_kernel_smoke(args)]
        print_table(rows, args.output_format)
        if args.json_output:
            write_json_output(args.json_output, rows)
        return

    old_checkpoint_cache_dir = configure_checkpoint_cache_dir(args)

    try:
        tokenizer_config = {"trust_remote_code": args.trust_remote_code}
        model, tokenizer = load(
            args.model,
            tokenizer_config=tokenizer_config,
            trust_remote_code=args.trust_remote_code,
        )

        rows = []
        if args.mode == "controlled-lcp":
            rows.extend(run_controlled_lcp(model, tokenizer, args))
        elif args.mode == "policy-sweep":
            rows.extend(run_policy_sweep(model, tokenizer, args))
        elif args.mode == "prefill-sweep":
            rows.extend(run_prefill_sweep(model, tokenizer, args))
        else:
            if args.mode == "queued":
                runner = run_queued_once
            elif args.mode == "batch":
                runner = run_batch_once
            else:
                runner = run_once
            if args.prompt_file:
                text = args.prompt_file.read_text()
                args.target_tokens = None
                for run in range(args.repeat_runs):
                    rows.extend(
                        run_with_from_q_a_kernel_sweep(
                            runner,
                            model,
                            tokenizer,
                            text,
                            args,
                            f"file-run-{run + 1}",
                        )
                    )
            elif args.mode == "queued":
                for run in range(args.repeat_runs):
                    rows.append(
                        runner(model, tokenizer, "", args, f"queued-run-{run + 1}")
                    )
            else:
                for length in parse_lengths(args.lengths):
                    args.target_tokens = length
                    text = build_prompt_text(tokenizer, length)
                    for run in range(args.repeat_runs):
                        rows.extend(
                            run_with_from_q_a_kernel_sweep(
                                runner,
                                model,
                                tokenizer,
                                text,
                                args,
                                f"synthetic-{length}",
                            )
                        )

                if args.mode == "single" and args.repeat_prefix_tokens > 0:
                    prefix = build_prompt_text(tokenizer, args.repeat_prefix_tokens)
                    for suffix in ("change-a", "change-b"):
                        args.target_tokens = (
                            args.repeat_prefix_tokens + args.repeat_suffix_tokens
                        )
                        text = (
                            f"{prefix}\n\n"
                            f"Repeated-prefix request {suffix}:\n"
                            "Reuse the stable prefix and analyze the changed suffix.\n"
                        )
                        snippet_id = suffix
                        while len(encode(tokenizer, text)) < args.target_tokens:
                            text += CODING_SNIPPET.format(i=snippet_id)
                            snippet_id = f"{snippet_id}-next"
                        for run in range(args.repeat_runs):
                            rows.extend(
                                run_with_from_q_a_kernel_sweep(
                                    runner,
                                    model,
                                    tokenizer,
                                    text,
                                    args,
                                    f"repeated-prefix-{suffix}",
                                )
                            )

        summaries = summarize_repeats(rows)
        print_table(summaries, args.output_format)
        if args.json_output:
            write_json_output(args.json_output, rows)
            if args.mode == "prefill-sweep":
                remove_partial_json_output(args.json_output)
    finally:
        restore_checkpoint_cache_dir(old_checkpoint_cache_dir)


if __name__ == "__main__":
    main()
