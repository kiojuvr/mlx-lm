#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "kernels/quantized_glm.h"

#define instantiate_quantized_head_flat(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      aligned,                                                                 \
      true)

#define instantiate_quantized_flat(name, type, group_size, bits, aligned)       \
  instantiate_kernel(                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      aligned)

#define instantiate_quantized_flat_tiled(                                       \
    name, type, group_size, bits, aligned, bm, bk, bn)                          \
  instantiate_kernel(                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned          \
      "_bm_" #bm "_bk_" #bk "_bn_" #bn,                                        \
      name,                                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      aligned,                                                                 \
      bm,                                                                      \
      bk,                                                                      \
      bn)

#define instantiate_quantized_head_broadcast(                                  \
    name, type, group_size, bits, aligned)                                      \
  instantiate_kernel(                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      aligned)

#define instantiate_moe_weighted_sum_tiled(type, score_type, topk, threads)    \
  instantiate_kernel(                                                          \
      "moe_weighted_sum_tiled_" #type "_score_" #score_type "_topk_" #topk     \
      "_t_" #threads,                                                          \
      moe_weighted_sum_tiled,                                                  \
      type,                                                                    \
      score_type,                                                              \
      topk,                                                                    \
      threads)

instantiate_quantized_head_flat(affine_qmm_t_head_flat, float16_t, 64, 8, true);
instantiate_quantized_head_flat(
    affine_qmm_t_head_flat,
    bfloat16_t,
    64,
    8,
    true);
instantiate_quantized_head_flat(affine_qmm_t_head_flat, float16_t, 64, 4, true);
instantiate_quantized_head_flat(
    affine_qmm_t_head_flat,
    bfloat16_t,
    64,
    4,
    true);
instantiate_quantized_head_broadcast(
    affine_qmm_t_head_broadcast,
    float16_t,
    64,
    4,
    true);
instantiate_quantized_head_broadcast(
    affine_qmm_t_head_broadcast,
    bfloat16_t,
    64,
    4,
    true);
instantiate_quantized_flat(affine_qmm_t_flat, float16_t, 64, 4, true);
instantiate_quantized_flat(affine_qmm_t_flat, bfloat16_t, 64, 4, true);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    float16_t,
    64,
    4,
    true,
    32,
    32,
    32);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    bfloat16_t,
    64,
    4,
    true,
    32,
    32,
    32);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    float16_t,
    64,
    4,
    true,
    32,
    64,
    32);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    bfloat16_t,
    64,
    4,
    true,
    32,
    64,
    32);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    float16_t,
    64,
    4,
    true,
    32,
    64,
    64);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    bfloat16_t,
    64,
    4,
    true,
    32,
    64,
    64);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    float16_t,
    64,
    4,
    true,
    64,
    64,
    32);
instantiate_quantized_flat_tiled(
    affine_qmm_t_flat_tiled,
    bfloat16_t,
    64,
    4,
    true,
    64,
    64,
    32);

instantiate_moe_weighted_sum_tiled(float16_t, float, 8, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 8, 256);
