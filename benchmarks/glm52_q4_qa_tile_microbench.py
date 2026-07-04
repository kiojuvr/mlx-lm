#!/usr/bin/env python3
"""Microbenchmark GLM-5.2 M3 q4 q_a projection tile variants.

This avoids loading the full model. It uses the fixed q_a projection shape
observed in GLM-5.2 M3: x[..., 6144] @ W[2048, 6144].T with affine q4 weights
and group size 64.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_lm.custom_kernels.glm_moe_dsa.fast import glm_dsa_q4_qa_proj_flat


DEFAULT_TILES = (
    "bk32",
    "bk64",
    "bm16",
    "bn16",
    "bn64",
    "bm64",
    "bm16bn64",
    "bm64bn64",
)

INPUT_DIM = 6144
OUTPUT_DIM = 2048
GROUP_SIZE = 64
BITS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--smoke-len", type=int, default=17)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=list(DEFAULT_TILES),
        choices=DEFAULT_TILES,
    )
    parser.add_argument(
        "--include-mlx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also benchmark mx.quantized_matmul as the baseline.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def seconds_for(fn, *, warmup_runs: int, runs: int) -> list[float]:
    for _ in range(warmup_runs):
        value = fn()
        mx.eval(value)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        value = fn()
        mx.eval(value)
        times.append(time.perf_counter() - start)
    return times


def summarize_times(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "best_seconds": min(times),
        "mean_seconds": sum(times) / len(times),
        "p50_seconds": ordered[len(ordered) // 2],
        "max_seconds": max(times),
    }


def diff_stats(value: mx.array, reference: mx.array) -> dict[str, float]:
    diff = mx.abs(value - reference)
    return {
        "max_abs_diff": float(mx.max(diff)),
        "mean_abs_diff": float(mx.mean(diff)),
    }


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.q_len < 1:
        raise ValueError("--q-len must be at least 1")
    if args.smoke_len < 1:
        raise ValueError("--smoke-len must be at least 1")

    rng = np.random.default_rng(args.seed)
    x = mx.array(
        rng.normal(
            0.0,
            0.1,
            size=(args.batch_size, args.q_len, INPUT_DIM),
        ).astype(np.float16)
    )
    weight = mx.array(
        rng.integers(
            0,
            2**32,
            size=(OUTPUT_DIM, INPUT_DIM // (32 // BITS)),
            dtype=np.uint32,
        )
    )
    scales = mx.array(
        rng.normal(
            0.0,
            0.02,
            size=(OUTPUT_DIM, INPUT_DIM // GROUP_SIZE),
        ).astype(np.float16)
    )
    biases = mx.array(
        rng.normal(
            0.0,
            0.02,
            size=(OUTPUT_DIM, INPUT_DIM // GROUP_SIZE),
        ).astype(np.float16)
    )
    mx.eval(x, weight, scales, biases)

    smoke_x = x[:, : min(args.smoke_len, args.q_len), :]
    dense_weight = mx.dequantize(
        weight,
        scales=scales,
        biases=biases,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode="affine",
    )
    reference = smoke_x @ dense_weight.T
    mx.eval(reference)

    rows: list[dict[str, Any]] = []
    if args.include_mlx:
        mlx_smoke = mx.quantized_matmul(
            smoke_x,
            weight,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
            mode="affine",
        )
        mx.eval(mlx_smoke)
        times = seconds_for(
            lambda: mx.quantized_matmul(
                x,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
                mode="affine",
            ),
            warmup_runs=args.warmup_runs,
            runs=args.runs,
        )
        rows.append(
            {
                "name": "mlx_quantized_matmul",
                "tile": "",
                **summarize_times(times),
                **diff_stats(mlx_smoke, reference),
            }
        )

    for tile in args.tiles:
        os.environ["MLX_LM_GLM_DSA_NATIVE_Q4_QA_TILE"] = tile
        smoke = glm_dsa_q4_qa_proj_flat(smoke_x, weight, scales, biases)
        mx.eval(smoke)
        times = seconds_for(
            lambda: glm_dsa_q4_qa_proj_flat(x, weight, scales, biases),
            warmup_runs=args.warmup_runs,
            runs=args.runs,
        )
        rows.append(
            {
                "name": "native_q4_qa",
                "tile": tile,
                **summarize_times(times),
                **diff_stats(smoke, reference),
            }
        )

    metadata = {
        "q_len": args.q_len,
        "batch_size": args.batch_size,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "smoke_len": min(args.smoke_len, args.q_len),
        "seed": args.seed,
    }
    result = {"metadata": metadata, "results": rows}

    print("name\ttile\tbest_s\tmean_s\tp50_s\tmax_abs\tmean_abs")
    for row in rows:
        print(
            "\t".join(
                [
                    row["name"],
                    row["tile"],
                    f"{row['best_seconds']:.6f}",
                    f"{row['mean_seconds']:.6f}",
                    f"{row['p50_seconds']:.6f}",
                    f"{row['max_abs_diff']:.8f}",
                    f"{row['mean_abs_diff']:.8g}",
                ]
            )
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
