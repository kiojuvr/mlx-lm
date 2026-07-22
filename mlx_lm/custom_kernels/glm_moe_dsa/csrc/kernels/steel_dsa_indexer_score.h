// Copyright © 2026 Apple Inc.

#pragma once

#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace mlx::steel;

constant bool do_causal [[function_constant(300)]];
constant bool weights_lh [[function_constant(301)]];
constant bool bucketed_topk_output [[function_constant(302)]];

template <typename T>
METAL_FUNC uint dsa_ordered_key_16(T x) {
  const ushort bits = as_type<ushort>(x);
  return (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
}

METAL_FUNC uint dsa_ordered_key_fp32(float x) {
  const uint bits = as_type<uint>(x);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template <typename T, typename O, int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void dsa_topk_indices_16bit(
    const device T* scores [[buffer(0)]],
    device O* out [[buffer(1)]],
    const constant DSATopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  threadgroup atomic_uint hist[256];
  threadgroup atomic_uint counters[2];
  threadgroup uint state[4];

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  if (tid < 2) {
    atomic_store_explicit(&counters[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* row_scores = scores + size_t(row) * params->K;
  device O* row_out = out + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    const int valid_length =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
    if (valid_length <= TOPK) {
      for (int i = int(tid); i < TOPK; i += THREADS) {
        row_out[i] = O(i < valid_length ? i : 0);
      }
      return;
    }
    scan_limit = valid_length;
  }

  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    atomic_fetch_add_explicit(&hist[key >> 8], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = 0;
    uint threshold_hi = 0;
    for (int h = 255; h >= 0; --h) {
      const uint count = atomic_load_explicit(&hist[h], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_hi = uint(h);
        break;
      }
      greater += count;
    }
    state[0] = threshold_hi;
    state[1] = greater;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_hi = state[0];
  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    if ((key >> 8) == threshold_hi) {
      atomic_fetch_add_explicit(&hist[key & 0xff], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = state[1];
    uint threshold_lo = 0;
    for (int l = 255; l >= 0; --l) {
      const uint count = atomic_load_explicit(&hist[l], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_lo = uint(l);
        break;
      }
      greater += count;
    }
    const uint threshold_key = (threshold_hi << 8) | threshold_lo;
    state[2] = threshold_key;
    state[3] = greater;
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[2];
  if (bucketed_topk_output) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = dsa_ordered_key_16(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = dsa_ordered_key_16(row_scores[i]);
      if (key > threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      } else if (key == threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      }
    }
  }
}

template <typename O, int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_topk_indices_fp32(
    const device float* scores [[buffer(0)]],
    device O* out [[buffer(1)]],
    const constant DSATopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  threadgroup atomic_uint hist[256];
  threadgroup atomic_uint counters[2];
  // Selected prefix, selected-prefix mask, and count greater than the prefix.
  threadgroup uint state[3];

  const device float* row_scores = scores + size_t(row) * params->K;
  device O* row_out = out + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    const int valid_length =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
    if (valid_length <= TOPK) {
      for (int i = int(tid); i < TOPK; i += THREADS) {
        row_out[i] = O(i < valid_length ? i : 0);
      }
      return;
    }
    scan_limit = valid_length;
  }

  if (tid == 0) {
    state[0] = 0;
    state[1] = 0;
    state[2] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Select the threshold key one byte at a time. This preserves score
  // differences below a half/bfloat ULP instead of rounding before top-k.
  for (int pass = 0; pass < 4; ++pass) {
    if (tid < 256) {
      atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int shift = 24 - pass * 8;
    const uint prefix = state[0];
    const uint prefix_mask = state[1];
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = dsa_ordered_key_fp32(row_scores[i]);
      if ((key & prefix_mask) == prefix) {
        atomic_fetch_add_explicit(
            &hist[(key >> shift) & 0xffu], 1, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      uint greater = state[2];
      uint threshold_byte = 0;
      for (int byte = 255; byte >= 0; --byte) {
        const uint count =
            atomic_load_explicit(&hist[byte], memory_order_relaxed);
        if (greater + count >= uint(TOPK)) {
          threshold_byte = uint(byte);
          break;
        }
        greater += count;
      }
      state[0] = prefix | (threshold_byte << shift);
      state[1] = prefix_mask | (0xffu << shift);
      state[2] = greater;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], state[2], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[0];
  if (bucketed_topk_output) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = dsa_ordered_key_fp32(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = dsa_ordered_key_fp32(row_scores[i]);
      if (key > threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      } else if (key == threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      }
    }
  }
}

template <
    typename QT,
    typename WT,
    typename OT,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
dsa_indexer_score(
    const device QT* Q [[buffer(0)]],
    const device QT* K [[buffer(1)]],
    const device WT* W [[buffer(2)]],
    device OT* O [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& H [[buffer(5)]],
    const constant int& unused_causal_prefix_topk [[buffer(6)]],
    const constant bool& skip_causal_future_store [[buffer(7)]],
    const constant int& causal_q_offset [[buffer(8)]],
    const constant float& scale [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  using gemm_kernel = GEMMKernel<
      QT,
      QT,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      true,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;
  constexpr int THREADS = WM * WN * 32;
  const int thread_idx = int(simd_group_id) * 32 + int(simd_lane_id);

  if (do_causal) {
    const int row_limit = metal::min(c_row + BM, M);
    if (c_col > q_offset + row_limit - 1) {
      if (skip_causal_future_store) {
        return;
      }
      device OT* Dst = O + size_t(tid.z) * M * N +
          size_t(c_row) * params->ldd + c_col;
      for (int e = thread_idx; e < BM * BN; e += THREADS) {
        const int row = e / BN;
        const int col = e - row * BN;
        if (c_row + row < M && c_col + col < N) {
          Dst[size_t(row) * params->ldd + col] =
              static_cast<OT>(-INFINITY);
        }
      }
      return;
    }
  }

  if (do_causal && unused_causal_prefix_topk > 0) {
    const int row_limit = metal::min(c_row + BM, M);
    if (q_offset + row_limit <= unused_causal_prefix_topk) {
      return;
    }
  }

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  O += size_t(tid.z) * M * N + size_t(c_row) * params->ldd + c_col;

  threadgroup QT As[gemm_kernel::tgp_mem_size_a];
  threadgroup QT Bs[gemm_kernel::tgp_mem_size_b];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    mma_op.Ctile.clear();

    const device QT* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device QT* B = K + size_t(c_col) * D;

    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
      const float weight = weights_lh
          ? static_cast<float>(W[size_t(row) * H + h])
          : static_cast<float>(W[size_t(h) * M + row]);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += max(frag[e] * scale, 0.0f) * weight;
        }
      }
    }
  }

  device OT* Dst = O + size_t(mma_op.sm) * params->ldd + mma_op.sn;
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      const int out_base =
          (i * decltype(mma_op.Ctile)::kFragRows) * WM * params->ldd +
          (j * decltype(mma_op.Ctile)::kFragCols) * WN;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        const bool future = do_causal && col > q_offset + row;
        const OT value = future ? static_cast<OT>(-INFINITY)
                                : static_cast<OT>(accum[ai]);
        Dst[out_base + e] = value;
        ai++;
      }
    }
  }
}

template <typename QT, typename WT, typename OT, int H, int D, int KEYS_PER_TG>
[[kernel, max_total_threads_per_threadgroup(KEYS_PER_TG * 32)]] void
dsa_indexer_score_decode(
    const device QT* Q [[buffer(0)]],
    const device QT* K [[buffer(1)]],
    const device WT* W [[buffer(2)]],
    device OT* O [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant float& scale [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  const int key = int(tid.x) * KEYS_PER_TG + int(simd_group_id);
  const int b = int(tid.y);
  if (key >= N) {
    return;
  }

  const device QT* q_base = Q + size_t(b) * H * D;
  const device QT* k_base = K + size_t(b) * N * D + size_t(key) * D;
  const device WT* w_base = W + size_t(b) * H;

  float score = 0.0f;
  STEEL_PRAGMA_UNROLL
  for (int h = 0; h < H; ++h) {
    float dot = 0.0f;
    STEEL_PRAGMA_UNROLL
    for (int d = int(simd_lane_id); d < D; d += 32) {
      dot += static_cast<float>(q_base[size_t(h) * D + d]) *
          static_cast<float>(k_base[d]);
    }
    dot = simd_sum(dot);
    if (simd_lane_id == 0) {
      score += max(dot * scale, 0.0f) * static_cast<float>(w_base[h]);
    }
  }

  if (simd_lane_id == 0) {
    O[size_t(b) * N + key] = static_cast<OT>(score);
  }
}
