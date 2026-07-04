#!/usr/bin/env python3
"""Microbenchmark GLM-5.2 M3 native sparse MLA tile variants.

This avoids loading the full model. It uses the fixed sparse MLA shape observed
in GLM-5.2 M3: q latent [B, 64, L, 512], q RoPE [B, 64, L, 64], latent KV
[B, 1, K, 512], RoPE K [B, 1, K, 64], and uint32 sparse top-k indices.
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

from mlx_lm.custom_kernels.glm_moe_dsa.fast import glm_dsa_sparse_mla_attention


DEFAULT_TILES = (
    "bk128",
    "bk256",
    "bk128_dc64",
    "wm4",
    "bk128_wm4",
    "bk128_dc64_wm4",
)

BATCH_SIZE = 1
HEADS = 64
LATENT_DIM = 512
PE_DIM = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=int, default=512)
    parser.add_argument("--k-len", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--smoke-q-len", type=int, default=16)
    parser.add_argument("--smoke-k-len", type=int, default=256)
    parser.add_argument("--smoke-topk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=list(DEFAULT_TILES),
        choices=DEFAULT_TILES,
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def synchronize(value: mx.array) -> None:
    mx.eval(value)
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def seconds_for(fn, *, warmup_runs: int, runs: int) -> list[float]:
    for _ in range(warmup_runs):
        synchronize(fn())
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        synchronize(fn())
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


def random_tensor(rng: np.random.Generator, shape: tuple[int, ...]) -> mx.array:
    return mx.array(rng.normal(0.0, 0.1, size=shape).astype(np.float16))


def make_inputs(
    rng: np.random.Generator,
    *,
    q_len: int,
    k_len: int,
    topk: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    q_latent = random_tensor(rng, (BATCH_SIZE, HEADS, q_len, LATENT_DIM))
    q_pe = random_tensor(rng, (BATCH_SIZE, HEADS, q_len, PE_DIM))
    kv_latent = random_tensor(rng, (BATCH_SIZE, 1, k_len, LATENT_DIM))
    k_pe = random_tensor(rng, (BATCH_SIZE, 1, k_len, PE_DIM))
    max_valid = max(1, k_len - q_len + 1)
    topk_indices = mx.array(
        rng.integers(
            0,
            max_valid,
            size=(BATCH_SIZE, 1, q_len, topk),
            dtype=np.uint32,
        )
    )
    mx.eval(q_latent, q_pe, kv_latent, k_pe, topk_indices)
    return q_latent, q_pe, kv_latent, k_pe, topk_indices


def make_full_topk(q_len: int, k_len: int, topk: int) -> mx.array:
    if topk > k_len:
        raise ValueError("--smoke-topk must be less than or equal to --smoke-k-len")
    indices = np.arange(topk, dtype=np.uint32)
    indices = np.broadcast_to(indices, (BATCH_SIZE, 1, q_len, topk)).copy()
    return mx.array(indices)


def dense_sparse_mla_reference(
    q_latent: mx.array,
    q_pe: mx.array,
    kv_latent: mx.array,
    k_pe: mx.array,
    scale: float,
) -> mx.array:
    q_len = q_latent.shape[2]
    k_len = kv_latent.shape[2]
    kv = mx.broadcast_to(kv_latent, (BATCH_SIZE, HEADS, k_len, LATENT_DIM))
    kp = mx.broadcast_to(k_pe, (BATCH_SIZE, HEADS, k_len, PE_DIM))
    scores = q_latent @ mx.swapaxes(kv, -1, -2)
    scores = scores + q_pe @ mx.swapaxes(kp, -1, -2)
    scores = scores * scale
    q_abs = mx.arange(k_len - q_len, k_len)[:, None]
    k_pos = mx.arange(k_len)[None, :]
    mask = k_pos <= q_abs
    scores = mx.where(mask[None, None, :, :], scores, -1e9)
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q_latent.dtype)
    return weights @ kv


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
    if args.q_len < 2:
        raise ValueError("--q-len must be at least 2")
    if args.k_len < args.q_len:
        raise ValueError("--k-len must be greater than or equal to --q-len")
    if args.topk < 16:
        raise ValueError("--topk must be at least 16")
    if args.smoke_q_len < 2:
        raise ValueError("--smoke-q-len must be at least 2")
    if args.smoke_k_len < args.smoke_q_len:
        raise ValueError(
            "--smoke-k-len must be greater than or equal to --smoke-q-len"
        )
    if args.smoke_topk < 16:
        raise ValueError("--smoke-topk must be at least 16")
    if args.smoke_topk != args.smoke_k_len:
        raise ValueError("--smoke-topk must equal --smoke-k-len for dense reference")
    if args.topk > args.k_len:
        raise ValueError("--topk must be less than or equal to --k-len")

    rng = np.random.default_rng(args.seed)
    scale = 1.0 / ((LATENT_DIM + PE_DIM) ** 0.5)
    smoke = make_inputs(
        rng,
        q_len=args.smoke_q_len,
        k_len=args.smoke_k_len,
        topk=args.smoke_topk,
    )
    smoke_topk = make_full_topk(args.smoke_q_len, args.smoke_k_len, args.smoke_topk)
    smoke = (*smoke[:4], smoke_topk)
    reference = dense_sparse_mla_reference(*smoke[:4], scale)
    synchronize(reference)

    perf_inputs = make_inputs(
        rng,
        q_len=args.q_len,
        k_len=args.k_len,
        topk=args.topk,
    )

    rows: list[dict[str, Any]] = []
    for tile in args.tiles:
        os.environ["MLX_LM_GLM_DSA_SPARSE_MLA_TILE"] = tile
        smoke_value = glm_dsa_sparse_mla_attention(*smoke, scale, causal=True)
        synchronize(smoke_value)
        times = seconds_for(
            lambda: glm_dsa_sparse_mla_attention(*perf_inputs, scale, causal=True),
            warmup_runs=args.warmup_runs,
            runs=args.runs,
        )
        rows.append(
            {
                "name": "native_sparse_mla",
                "tile": tile,
                **summarize_times(times),
                **diff_stats(smoke_value, reference),
            }
        )

    metadata = {
        "q_len": args.q_len,
        "k_len": args.k_len,
        "topk": args.topk,
        "smoke_q_len": args.smoke_q_len,
        "smoke_k_len": args.smoke_k_len,
        "smoke_topk": args.smoke_topk,
        "batch_size": BATCH_SIZE,
        "heads": HEADS,
        "latent_dim": LATENT_DIM,
        "pe_dim": PE_DIM,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
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
