#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <vector>

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array glm_dsa_q8_vup_flat(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q4_vup_flat(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q4_qa_proj_flat(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q4_qb_proj_flat(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q4_qb_proj_heads(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

std::vector<mx::array> glm_dsa_q4_qb_proj_split(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q_a_rms_norm(
    const mx::array& x,
    const mx::array& weight,
    float eps,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q_a_rms_scale(
    const mx::array& x,
    float eps,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q4_qb_proj_scaled_heads(
    const mx::array& x,
    const mx::array& norm_weight,
    const mx::array& row_scales,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_dsa_q4_qb_proj_wscaled_heads(
    const mx::array& x,
    const mx::array& norm_weight,
    const mx::array& row_scales,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    mx::StreamOrDevice s = {});

mx::array glm_moe_weighted_sum(
    const mx::array& x_sorted,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
