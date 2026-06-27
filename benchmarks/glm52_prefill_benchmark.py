#!/usr/bin/env python3
"""Lightweight GLM-5.2 prefill benchmark for long coding prompts."""

import argparse
import csv
import json
import logging
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


def parse_lengths(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


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


def extract_checkpoint_summary(messages):
    summary = {
        "checkpoint_total_prompt_tokens": None,
        "server_cached_tokens": None,
        "disk_cached_tokens": None,
        "fresh_prompt_tokens": None,
        "fresh_prefill_tokens": None,
        "checkpoint_prefill_step_size": None,
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
        if "prefill summary " not in message:
            continue
        for key, value in re.findall(r"([a-z_]+)=([^ ]+)", message):
            output_key = {
                "total_prompt_tokens": "checkpoint_total_prompt_tokens",
                "prefill_step_size": "checkpoint_prefill_step_size",
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
            else:
                summary[output_key] = int(value)
    return summary


def configure_glm_dsa_fast_prefill(args):
    if args.fast_prefill == "enabled":
        os.environ[glm_moe_dsa.GLM_DSA_FAST_PREFILL_ENV] = "1"
    elif args.fast_prefill == "disabled":
        os.environ[glm_moe_dsa.GLM_DSA_FAST_PREFILL_ENV] = "0"
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
    if args.prefill_profile:
        os.environ[glm_moe_dsa.GLM_DSA_PREFILL_PROFILE_ENV] = "1"


def reset_glm_dsa_profile():
    glm_moe_dsa.reset_glm_dsa_prefill_profile()


def collect_glm_dsa_profile(args):
    profile = glm_moe_dsa.get_glm_dsa_prefill_profile()
    stage_values = {}
    for stage, values in profile["stages"].items():
        key = f"glm_dsa_{stage}_seconds"
        stage_values[key] = values["seconds"] if args.prefill_profile else None
        stage_values[f"glm_dsa_{stage}_count"] = values["count"]
    return {
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
        "glm_dsa_fast_prefill_hits": profile["fast_prefill_hits"],
        "glm_dsa_fast_prefill_fallback_reasons": profile["fallback_reasons"],
        **stage_values,
    }


def run_once(model, tokenizer, prompt, args, case_name):
    tokenize_t0 = time.perf_counter()
    tokens = prompt_to_tokens(tokenizer, prompt)
    if args.target_tokens is not None:
        tokens = tokens[: args.target_tokens]
    tokenize_seconds = time.perf_counter() - tokenize_t0

    progress_events = []
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

    ttft_t0 = time.perf_counter()
    response = None
    try:
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=tokens,
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            prompt_checkpoint=not args.no_prompt_checkpoint,
            prompt_checkpoint_store_prefix_lengths=args.checkpoint_store_prefix_lengths,
            prompt_checkpoint_save_exact=(
                args.checkpoint_save_exact == "enabled"
            ),
            prompt_checkpoint_frontier_min_tokens=(
                args.checkpoint_frontier_min_tokens
            ),
            prompt_checkpoint_frontier_stride_tokens=(
                args.checkpoint_frontier_stride_tokens
            ),
            prompt_progress_callback=lambda done, total: progress_events.append(
                (done, total, time.perf_counter())
            ),
        ):
            break
        mx.synchronize()
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
        if old_debug is None:
            os.environ.pop(PROMPT_CHECKPOINT_DEBUG_ENV, None)
        else:
            os.environ[PROMPT_CHECKPOINT_DEBUG_ENV] = old_debug

    ttft_seconds = time.perf_counter() - ttft_t0
    if response is None:
        raise RuntimeError(f"{case_name}: generation produced no response")

    checkpoint = extract_checkpoint_summary(capture.messages)
    glm_profile = collect_glm_dsa_profile(args)
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
        "prompt_tps": response.prompt_tps,
        "peak_memory_gb": response.peak_memory,
        "progress_events": len(progress_events),
        "finish_reason": response.finish_reason,
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
        "checkpoint_cache_dir",
        "checkpoint_save_exact",
        "checkpoint_lookup_seconds",
        "checkpoint_files_scanned",
        "checkpoint_candidates_scanned",
        "checkpoint_matched_candidates",
        "checkpoint_manifest_entries",
        "checkpoint_manifest_bootstrap",
        "prompt_tokens",
        "tokenize_seconds",
        "ttft_seconds",
        "ttft_p50_seconds",
        "ttft_p95_seconds",
        "prompt_tps",
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
        "glm_dsa_fast_prefill",
        "glm_dsa_fast_prefill_env",
        "glm_dsa_fast_prefill_query_chunk",
        "glm_dsa_fast_prefill_key_block",
        "glm_dsa_sparse_prefill_min_context",
        "glm_dsa_fast_prefill_hits",
        "glm_dsa_fast_prefill_fallback_reasons",
        "glm_dsa_q_projection_seconds",
        "glm_dsa_kv_cache_update_seconds",
        "glm_dsa_dsa_indexer_topk_seconds",
        "glm_dsa_latent_kv_dequantization_seconds",
        "glm_dsa_latent_kv_projection_seconds",
        "glm_dsa_sparse_gather_seconds",
        "glm_dsa_attention_seconds",
        "glm_dsa_total_prefill_seconds",
    ]
    delimiter = "," if output_format == "csv" else "\t"
    writer = csv.writer(sys.stdout, delimiter=delimiter, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow([format_output_cell(row.get(h)) for h in headers])


def configure_checkpoint_cache_dir(args):
    if args.mode == "controlled-lcp" and args.checkpoint_cache_dir is None:
        args.checkpoint_cache_dir = Path(
            tempfile.mkdtemp(prefix="glm52-lcp-checkpoints-")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or HF repo.")
    parser.add_argument(
        "--mode",
        choices=("single", "batch", "queued", "controlled-lcp"),
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
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
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
        "--prefill-profile",
        action="store_true",
        help=(
            "Synchronize and report GLM DSA prefill stage timings. This adds "
            "profiling overhead and is intended for measurement runs."
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
    if args.checkpoint_save_exact is None:
        args.checkpoint_save_exact = (
            "disabled" if args.mode == "controlled-lcp" else "enabled"
        )
    args.target_tokens = None
    configure_glm_dsa_fast_prefill(args)
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
                    rows.append(
                        runner(model, tokenizer, text, args, f"file-run-{run + 1}")
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
                        rows.append(
                            runner(
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
                            rows.append(
                                runner(
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
            args.json_output.write_text(
                json.dumps({"runs": rows, "summary": summaries}, indent=2)
            )
    finally:
        restore_checkpoint_cache_dir(old_checkpoint_cache_dir)


if __name__ == "__main__":
    main()
