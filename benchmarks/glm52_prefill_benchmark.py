#!/usr/bin/env python3
"""Lightweight GLM-5.2 prefill benchmark for long coding prompts."""

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm.generate import PROMPT_CHECKPOINT_DEBUG_ENV, BatchGenerator, BatchStats
from mlx_lm.generate import stream_generate
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
        "checkpoint_resolution": None,
        "checkpoint_lookup_seconds": None,
        "checkpoint_events": len(messages),
    }
    for message in messages:
        if "prefill summary " not in message:
            continue
        resolution = re.search(r"resolution=([^ ]+)", message)
        lookup_seconds = re.search(r"lookup_seconds=([0-9.]+)", message)
        if resolution:
            summary["checkpoint_resolution"] = resolution.group(1)
        if lookup_seconds:
            summary["checkpoint_lookup_seconds"] = float(lookup_seconds.group(1))
    return summary


def run_once(model, tokenizer, text, args, case_name):
    tokenize_t0 = time.perf_counter()
    tokens = encode(tokenizer, text)
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
    return {
        "case": case_name,
        "mode": "single",
        "batch_size": 1,
        "prompt_tokens": len(tokens),
        "tokenize_seconds": tokenize_seconds,
        "ttft_seconds": ttft_seconds,
        "prompt_tps": response.prompt_tps,
        "peak_memory_gb": response.peak_memory,
        "progress_events": len(progress_events),
        "finish_reason": response.finish_reason,
        **checkpoint,
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
        "checkpoint_lookup_seconds": None,
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
        "checkpoint_lookup_seconds": None,
        "checkpoint_events": 0,
        "admission_wait_p50_seconds": percentile(wait_seconds, 50),
        "admission_wait_p95_seconds": percentile(wait_seconds, 95),
        "active_batch_size_max": max(active_sizes) if active_sizes else 0,
        **admission_stats,
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


def print_table(rows):
    headers = [
        "case",
        "mode",
        "batch_size",
        "prefill_batch_size",
        "completion_batch_size",
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
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                str(round(row[h], 4))
                if isinstance(row.get(h), float)
                else str(row.get(h))
                for h in headers
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or HF repo.")
    parser.add_argument(
        "--mode", choices=("single", "batch", "queued"), default="single"
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
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--kv-bits", type=int)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-prompt-checkpoint", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    tokenizer_config = {"trust_remote_code": args.trust_remote_code}
    model, tokenizer = load(
        args.model,
        tokenizer_config=tokenizer_config,
        trust_remote_code=args.trust_remote_code,
    )

    rows = []
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
            rows.append(runner(model, tokenizer, text, args, f"file-run-{run + 1}"))
    elif args.mode == "queued":
        for run in range(args.repeat_runs):
            rows.append(runner(model, tokenizer, "", args, f"queued-run-{run + 1}"))
    else:
        for length in parse_lengths(args.lengths):
            args.target_tokens = length
            text = build_prompt_text(tokenizer, length)
            for run in range(args.repeat_runs):
                rows.append(
                    runner(model, tokenizer, text, args, f"synthetic-{length}")
                )

        if args.mode == "single" and args.repeat_prefix_tokens > 0:
            prefix = build_prompt_text(tokenizer, args.repeat_prefix_tokens)
            for suffix in ("change-a", "change-b"):
                args.target_tokens = args.repeat_prefix_tokens + 256
                text = prefix + CODING_SNIPPET.format(i=suffix)
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
    print_table(summaries)
    if args.json_output:
        args.json_output.write_text(
            json.dumps({"runs": rows, "summary": summaries}, indent=2)
        )


if __name__ == "__main__":
    main()
