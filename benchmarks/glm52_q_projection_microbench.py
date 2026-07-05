#!/usr/bin/env python3
"""Microbenchmark GLM-5.2 M3 q_projection structure variants.

This avoids loading the full model. It composes the fixed GLM-5.2 M3 q_a and
q_b projection shapes:

  q_a: x[..., 6144] @ W[2048, 6144].T
  q_b: qr[..., 2048] @ W[16384, 2048].T -> [B, 64, L, 256]

The goal is to compare whole q_projection structure costs, including RMSNorm,
q_b output layout, q_nope/q_pe splitting, and the shared-layer q_b-from-q_a
materialization-avoidance kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from mlx_lm.custom_kernels.glm_moe_dsa.fast import (
    glm_dsa_q_a_rms_norm,
    glm_dsa_q_a_rms_scale,
    glm_dsa_q4_qa_proj_flat,
    glm_dsa_q4_qb_proj_flat,
    glm_dsa_q4_qb_proj_heads,
    glm_dsa_q4_qb_proj_split,
    glm_dsa_q4_qb_proj_scaled_heads,
    glm_dsa_q4_qb_proj_wscaled_heads,
)


Q_INPUT_DIM = 6144
Q_LORA_RANK = 2048
HEADS = 64
HEAD_DIM = 256
Q_OUTPUT_DIM = HEADS * HEAD_DIM
QK_NOPE_HEAD_DIM = 128
GROUP_SIZE = 64
BITS = 4

DEFAULT_CANDIDATES = (
    "baseline_mlx",
    "native_qa_mlx_qb",
    "native_qa_native_qb_flat",
    "native_qa_native_qb_flat_split",
    "native_qa_native_qb_heads",
    "native_qa_native_qb_split",
    "native_qa_native_rms_native_qb_heads",
    "native_qa_from_q_a_scaled",
    "native_qa_from_q_a_wscaled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--smoke-len", type=int, default=17)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--qa-tile", default="bk64")
    parser.add_argument("--qb-tile", default="bm64")
    parser.add_argument("--scaled-tile", default="bn64")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
        choices=DEFAULT_CANDIDATES,
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def collect_arrays(value: Any, arrays: list[mx.array]) -> None:
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            collect_arrays(item, arrays)
    elif isinstance(value, dict):
        for item in value.values():
            collect_arrays(item, arrays)


def synchronize(value: Any) -> None:
    arrays: list[mx.array] = []
    collect_arrays(value, arrays)
    if arrays:
        mx.eval(*arrays)
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def seconds_for(
    fn: Callable[[], Any],
    *,
    warmup_runs: int,
    runs: int,
) -> list[float]:
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


def output_heads(value: tuple[mx.array, mx.array]) -> mx.array:
    q_nope, q_pe = value
    return mx.concatenate([q_nope, q_pe], axis=-1)


def diff_stats(value: tuple[mx.array, mx.array], reference: tuple[mx.array, mx.array]):
    diff = mx.abs(output_heads(value) - output_heads(reference))
    return {
        "max_abs_diff": float(mx.max(diff)),
        "mean_abs_diff": float(mx.mean(diff)),
    }


def random_affine_q4(
    rng: np.random.Generator,
    *,
    output_dim: int,
    input_dim: int,
) -> tuple[mx.array, mx.array, mx.array]:
    weight = mx.array(
        rng.integers(
            0,
            2**32,
            size=(output_dim, input_dim // (32 // BITS)),
            dtype=np.uint32,
        )
    )
    scales = mx.array(
        rng.normal(
            0.0,
            0.02,
            size=(output_dim, input_dim // GROUP_SIZE),
        ).astype(np.float16)
    )
    biases = mx.array(
        rng.normal(
            0.0,
            0.02,
            size=(output_dim, input_dim // GROUP_SIZE),
        ).astype(np.float16)
    )
    return weight, scales, biases


class QProjectionBench:
    def __init__(self, args: argparse.Namespace):
        rng = np.random.default_rng(args.seed)
        self.args = args
        self.x = mx.array(
            rng.normal(
                0.0,
                0.1,
                size=(args.batch_size, args.q_len, Q_INPUT_DIM),
            ).astype(np.float16)
        )
        self.q_a_weight, self.q_a_scales, self.q_a_biases = random_affine_q4(
            rng,
            output_dim=Q_LORA_RANK,
            input_dim=Q_INPUT_DIM,
        )
        self.q_b_weight, self.q_b_scales, self.q_b_biases = random_affine_q4(
            rng,
            output_dim=Q_OUTPUT_DIM,
            input_dim=Q_LORA_RANK,
        )
        self.norm_weight = mx.array(
            rng.normal(1.0, 0.02, size=(Q_LORA_RANK,)).astype(np.float16)
        )
        synchronize(
            (
                self.x,
                self.q_a_weight,
                self.q_a_scales,
                self.q_a_biases,
                self.q_b_weight,
                self.q_b_scales,
                self.q_b_biases,
                self.norm_weight,
            )
        )

    def split_heads(self, q_heads: mx.array) -> tuple[mx.array, mx.array]:
        q_nope, q_pe = mx.split(q_heads, [QK_NOPE_HEAD_DIM], axis=-1)
        return q_nope, q_pe

    def q_b_flat_to_heads(self, q_flat: mx.array) -> mx.array:
        B, L, _ = q_flat.shape
        return q_flat.reshape(B, L, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    def split_flat_before_transpose(
        self,
        q_flat: mx.array,
    ) -> tuple[mx.array, mx.array]:
        B, L, _ = q_flat.shape
        q_view = q_flat.reshape(B, L, HEADS, HEAD_DIM)
        q_nope = q_view[..., :QK_NOPE_HEAD_DIM].transpose(0, 2, 1, 3)
        q_pe = q_view[..., QK_NOPE_HEAD_DIM:].transpose(0, 2, 1, 3)
        return q_nope, q_pe

    def mlx_q_a(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x,
            self.q_a_weight,
            scales=self.q_a_scales,
            biases=self.q_a_biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
            mode="affine",
        )

    def native_q_a(self, x: mx.array) -> mx.array:
        return glm_dsa_q4_qa_proj_flat(
            x,
            self.q_a_weight,
            self.q_a_scales,
            self.q_a_biases,
        )

    def mlx_q_b_flat(self, qr: mx.array) -> mx.array:
        return mx.quantized_matmul(
            qr,
            self.q_b_weight,
            scales=self.q_b_scales,
            biases=self.q_b_biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
            mode="affine",
        )

    def native_q_b_flat(self, qr: mx.array) -> mx.array:
        return glm_dsa_q4_qb_proj_flat(
            qr,
            self.q_b_weight,
            self.q_b_scales,
            self.q_b_biases,
        )

    def native_q_b_heads(self, qr: mx.array) -> mx.array:
        return glm_dsa_q4_qb_proj_heads(
            qr,
            self.q_b_weight,
            self.q_b_scales,
            self.q_b_biases,
        )

    def native_q_b_split(self, qr: mx.array) -> tuple[mx.array, mx.array]:
        return glm_dsa_q4_qb_proj_split(
            qr,
            self.q_b_weight,
            self.q_b_scales,
            self.q_b_biases,
        )

    def rms_norm(self, q_a: mx.array) -> mx.array:
        return mx.fast.rms_norm(q_a, self.norm_weight, self.args.eps)

    def native_rms_norm(self, q_a: mx.array) -> mx.array:
        return glm_dsa_q_a_rms_norm(q_a, self.norm_weight, self.args.eps)

    def baseline_mlx(self, x: mx.array) -> tuple[mx.array, mx.array]:
        q_a = self.mlx_q_a(x)
        qr = self.rms_norm(q_a)
        q_flat = self.mlx_q_b_flat(qr)
        return self.split_heads(self.q_b_flat_to_heads(q_flat))

    def native_qa_mlx_qb(self, x: mx.array) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        qr = self.rms_norm(q_a)
        q_flat = self.mlx_q_b_flat(qr)
        return self.split_heads(self.q_b_flat_to_heads(q_flat))

    def native_qa_native_qb_flat(self, x: mx.array) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        qr = self.rms_norm(q_a)
        q_flat = self.native_q_b_flat(qr)
        return self.split_heads(self.q_b_flat_to_heads(q_flat))

    def native_qa_native_qb_flat_split(
        self,
        x: mx.array,
    ) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        qr = self.rms_norm(q_a)
        q_flat = self.native_q_b_flat(qr)
        return self.split_flat_before_transpose(q_flat)

    def native_qa_native_qb_heads(self, x: mx.array) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        qr = self.rms_norm(q_a)
        return self.split_heads(self.native_q_b_heads(qr))

    def native_qa_native_qb_split(self, x: mx.array) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        qr = self.rms_norm(q_a)
        return self.native_q_b_split(qr)

    def native_qa_native_rms_native_qb_heads(
        self,
        x: mx.array,
    ) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        qr = self.native_rms_norm(q_a)
        return self.split_heads(self.native_q_b_heads(qr))

    def native_qa_from_q_a_scaled(self, x: mx.array) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        row_scales = glm_dsa_q_a_rms_scale(q_a, self.args.eps)
        q_heads = glm_dsa_q4_qb_proj_scaled_heads(
            q_a,
            self.norm_weight,
            row_scales,
            self.q_b_weight,
            self.q_b_scales,
            self.q_b_biases,
        )
        return self.split_heads(q_heads)

    def native_qa_from_q_a_wscaled(self, x: mx.array) -> tuple[mx.array, mx.array]:
        q_a = self.native_q_a(x)
        row_scales = glm_dsa_q_a_rms_scale(q_a, self.args.eps)
        q_heads = glm_dsa_q4_qb_proj_wscaled_heads(
            q_a,
            self.norm_weight,
            row_scales,
            self.q_b_weight,
            self.q_b_scales,
            self.q_b_biases,
        )
        return self.split_heads(q_heads)


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.q_len < 1:
        raise ValueError("--q-len must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.smoke_len < 1:
        raise ValueError("--smoke-len must be at least 1")

    os.environ["MLX_LM_GLM_DSA_NATIVE_Q4_QA_TILE"] = args.qa_tile
    os.environ["MLX_LM_GLM_DSA_NATIVE_Q4_QB_TILE"] = args.qb_tile
    os.environ["MLX_LM_GLM_DSA_NATIVE_Q4_QB_SCALED_TILE"] = args.scaled_tile

    bench = QProjectionBench(args)
    smoke_x = bench.x[:, : min(args.smoke_len, args.q_len), :]
    reference = bench.baseline_mlx(smoke_x)
    synchronize(reference)

    rows: list[dict[str, Any]] = []
    for candidate in args.candidates:
        fn = getattr(bench, candidate)
        smoke = fn(smoke_x)
        synchronize(smoke)
        times = seconds_for(
            lambda fn=fn: fn(bench.x),
            warmup_runs=args.warmup_runs,
            runs=args.runs,
        )
        rows.append(
            {
                "name": candidate,
                **summarize_times(times),
                **diff_stats(smoke, reference),
            }
        )

    metadata = {
        "q_len": args.q_len,
        "batch_size": args.batch_size,
        "q_input_dim": Q_INPUT_DIM,
        "q_lora_rank": Q_LORA_RANK,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "q_output_dim": Q_OUTPUT_DIM,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "smoke_len": min(args.smoke_len, args.q_len),
        "eps": args.eps,
        "seed": args.seed,
        "qa_tile": args.qa_tile,
        "qb_tile": args.qb_tile,
        "scaled_tile": args.scaled_tile,
    }
    result = {"metadata": metadata, "results": rows}

    print("name\tbest_s\tmean_s\tp50_s\tmax_abs\tmean_abs")
    for row in rows:
        print(
            "\t".join(
                [
                    row["name"],
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
