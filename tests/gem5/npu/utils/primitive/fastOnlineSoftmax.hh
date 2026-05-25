#ifndef TESTS_GEM5_NPU_UTILS_PRIMITIVE_FASTONLINESOFTMAX_HH_
#define TESTS_GEM5_NPU_UTILS_PRIMITIVE_FASTONLINESOFTMAX_HH_

#include <stdint.h>

#include <cstdio>

#include "internal.hh"

typedef struct
{
    uint32_t device_id;
    uint32_t sync_indicator;
    uint32_t set_completion_sync;
    uint32_t vpu_dlen_bytes;
    uint64_t port_base;
} NpuFastOnlineSoftmaxLaunchConfig;

typedef struct
{
    uint32_t template_build_count;
    uint32_t launched_cmd_count;
    uint32_t slice_count;
} NpuFastOnlineSoftmaxStats;

typedef struct
{
    NpuCmd reduce_max;
    NpuCmd sub_max_delta;
    NpuCmd abs_max_delta;
    NpuCmd add_max_sum;
    NpuCmd add_max_merge;
    NpuCmd scale_m_next;
    NpuCmd sub_scores;
    NpuCmd exp_scores;
    NpuCmd reduce_sum;
    NpuCmd sub_prev;
    NpuCmd exp_prev;
    NpuCmd mul_prev_l;
    NpuCmd add_l;
    NpuCmd div_prob;
} NpuFastOnlineSoftmaxCmdTemplates;

enum
{
    NPU_FAST_ONLINE_SOFTMAX_VECTOR_SCRATCH_COUNT = 8U,
    NPU_FAST_ONLINE_SOFTMAX_MATRIX_SCRATCH_COUNT = 2U,
    NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT = 14U,
};

static inline int
npu_fast_online_softmax_validate_launch_config(
    const NpuFastOnlineSoftmaxLaunchConfig *config)
{
    if (config == NULL) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=null_launch_config\n");
        return -1;
    }
    if (config->set_completion_sync != 0U && config->sync_indicator == 0U) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=missing_sync_indicator\n");
        return -1;
    }
    if (config->vpu_dlen_bytes == 0U) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=missing_vpu_dlen\n");
        return -1;
    }
    return 0;
}

static inline int
npu_fast_online_softmax_validate_layout_size(
    const char *label, const PrimitiveTensorDesc &tensor,
    uint32_t required_dlen_bytes)
{
    const PrimitiveTensorDesc tensor2d = primitivePeelTo2D(tensor, 0U);
    const uint32_t elem_bytes = tensor2d.elemBytes();

    if (elem_bytes == 0U || (PRIMITIVE_DLEN_BYTES % elem_bytes) != 0U ||
        (required_dlen_bytes % elem_bytes) != 0U) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=%s_invalid_layout_unit\n",
               label);
        return -1;
    }

    const uint32_t primitive_layout_size = tensor2d.layoutSizeElems != 0U ?
        tensor2d.layoutSizeElems : (PRIMITIVE_DLEN_BYTES / elem_bytes);
    const uint32_t required_layout_size = required_dlen_bytes / elem_bytes;
    if (primitive_layout_size != required_layout_size) {
        printf(
            "FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=%s_layout_size_mismatch "
            "primitive=%u required=%u\n",
            label, primitive_layout_size, required_layout_size);
        return -1;
    }

    return 0;
}

static inline int
npu_fast_online_softmax_validate_tensor_descs(
    const PrimitiveTensorDesc &scores_block, const PrimitiveTensorDesc &m_prev,
    const PrimitiveTensorDesc &l_prev, const PrimitiveTensorDesc &m_next,
    const PrimitiveTensorDesc &l_next, const PrimitiveTensorDesc &p_block,
    uint32_t required_dlen_bytes)
{
    const PrimitiveTensorDesc scores2d = primitivePeelTo2D(scores_block, 0U);
    const PrimitiveTensorDesc m_prev2d = primitivePeelTo2D(m_prev, 0U);
    const PrimitiveTensorDesc l_prev2d = primitivePeelTo2D(l_prev, 0U);
    const PrimitiveTensorDesc m_next2d = primitivePeelTo2D(m_next, 0U);
    const PrimitiveTensorDesc l_next2d = primitivePeelTo2D(l_next, 0U);
    const PrimitiveTensorDesc p_block2d = primitivePeelTo2D(p_block, 0U);

    if (scores_block.sliceCount(2U) != 1U || p_block.sliceCount(2U) != 1U ||
        m_prev.sliceCount(2U) != 1U || l_prev.sliceCount(2U) != 1U ||
        m_next.sliceCount(2U) != 1U || l_next.sliceCount(2U) != 1U) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=unsupported_slice_count\n");
        return -1;
    }
    if (scores2d.dataType != VPU_DATA_F32 ||
        m_prev2d.dataType != VPU_DATA_F32 ||
        l_prev2d.dataType != VPU_DATA_F32 ||
        m_next2d.dataType != VPU_DATA_F32 ||
        l_next2d.dataType != VPU_DATA_F32 ||
        p_block2d.dataType != VPU_DATA_F32) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=unsupported_dtype\n");
        return -1;
    }
    if (scores2d.rank() != 2U || p_block2d.rank() != 2U ||
        m_prev2d.rank() != 2U || l_prev2d.rank() != 2U ||
        m_next2d.rank() != 2U || l_next2d.rank() != 2U) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=unexpected_rank\n");
        return -1;
    }
    if (scores2d.dim(0) == 0U || scores2d.dim(1) == 0U) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=empty_scores\n");
        return -1;
    }
    if (p_block2d.dim(0) != scores2d.dim(0) ||
        p_block2d.dim(1) != scores2d.dim(1)) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=prob_shape_mismatch\n");
        return -1;
    }
    if (m_prev2d.dim(0) != scores2d.dim(0) || m_prev2d.dim(1) != 1U ||
        l_prev2d.dim(0) != scores2d.dim(0) || l_prev2d.dim(1) != 1U ||
        m_next2d.dim(0) != scores2d.dim(0) || m_next2d.dim(1) != 1U ||
        l_next2d.dim(0) != scores2d.dim(0) || l_next2d.dim(1) != 1U) {
        printf("FAST_ONLINE_SOFTMAX_VALIDATE_FAIL=state_shape_mismatch\n");
        return -1;
    }
    if (npu_fast_online_softmax_validate_layout_size(
            "scores", scores_block, required_dlen_bytes) != 0 ||
        npu_fast_online_softmax_validate_layout_size(
            "m_prev", m_prev, required_dlen_bytes) != 0 ||
        npu_fast_online_softmax_validate_layout_size(
            "l_prev", l_prev, required_dlen_bytes) != 0 ||
        npu_fast_online_softmax_validate_layout_size(
            "m_next", m_next, required_dlen_bytes) != 0 ||
        npu_fast_online_softmax_validate_layout_size(
            "l_next", l_next, required_dlen_bytes) != 0 ||
        npu_fast_online_softmax_validate_layout_size(
            "p_block", p_block, required_dlen_bytes) != 0) {
        return -1;
    }

    return 0;
}

static inline uint32_t
npu_fast_online_softmax_scratch_slot_span(
    const PrimitiveTensorDesc &scores_block)
{
    const PrimitiveTensorDesc scores2d = primitivePeelTo2D(scores_block, 0U);
    const PrimitiveTensorDesc vector_like =
        primitivePackedReduceLike(0U, scores2d);
    const PrimitiveTensorDesc matrix_like = primitivePackedLike(0U, scores2d);

    return primitiveTensorSlotSpan(vector_like) *
               NPU_FAST_ONLINE_SOFTMAX_VECTOR_SCRATCH_COUNT +
           primitiveTensorSlotSpan(matrix_like) *
               NPU_FAST_ONLINE_SOFTMAX_MATRIX_SCRATCH_COUNT;
}

static inline void
npu_fast_online_softmax_set_sync(NpuCmd *cmd,
                                 const NpuFastOnlineSoftmaxLaunchConfig
                                     *config,
                                 bool set_completion_sync)
{
    if (config->sync_indicator == 0U && !set_completion_sync) {
        return;
    }

    PrimitiveSyncDesc sync = {};
    sync.syncIndicator = config->sync_indicator;
    sync.setSnsIndicator =
        set_completion_sync && config->set_completion_sync != 0U;
    primitiveSetSync(cmd, sync);
}

static inline void
npu_fast_online_softmax_build_cmd_templates(
    const PrimitiveVpu2DDesc &scores_vpu,
    const PrimitiveVpu2DDesc &m_prev_vpu,
    const PrimitiveVpu2DDesc &l_prev_vpu,
    const PrimitiveVpu2DDesc &row_max_vpu,
    const PrimitiveVpu2DDesc &max_delta_vpu,
    const PrimitiveVpu2DDesc &max_abs_vpu,
    const PrimitiveVpu2DDesc &max_sum_vpu,
    const PrimitiveVpu2DDesc &m_next_vpu,
    const PrimitiveVpu2DDesc &shifted_vpu,
    const PrimitiveVpu2DDesc &exp_block_vpu,
    const PrimitiveVpu2DDesc &block_sum_vpu,
    const PrimitiveVpu2DDesc &prev_delta_vpu,
    const PrimitiveVpu2DDesc &prev_scale_vpu,
    const PrimitiveVpu2DDesc &prev_scaled_l_vpu,
    const PrimitiveVpu2DDesc &l_next_vpu,
    const PrimitiveVpu2DDesc &p_block_vpu,
    const NpuFastOnlineSoftmaxLaunchConfig *config,
    NpuFastOnlineSoftmaxCmdTemplates *templates)
{
    const uint32_t half_scale = primitiveFloatToBits(0.5f);

    templates->reduce_max = primitiveBuildReduceExecCmd(
        config->device_id, VPU_OP_VREDUCE_MAX, scores_vpu, row_max_vpu);
    templates->sub_max_delta = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VSUB, m_prev_vpu, row_max_vpu,
        max_delta_vpu);
    templates->abs_max_delta = primitiveBuildUnaryExecCmd(
        config->device_id, VPU_OP_VABS, max_delta_vpu, max_abs_vpu, 0U);
    templates->add_max_sum = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VADD, m_prev_vpu, row_max_vpu,
        max_sum_vpu);
    templates->add_max_merge = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VADD, max_sum_vpu, max_abs_vpu, m_next_vpu);
    templates->scale_m_next = primitiveBuildUnaryExecCmd(
        config->device_id, VPU_OP_VSCALE, m_next_vpu, m_next_vpu, half_scale);
    templates->sub_scores = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VSUB, scores_vpu, m_next_vpu, shifted_vpu);
    templates->exp_scores = primitiveBuildUnaryExecCmd(
        config->device_id, VPU_OP_VEXP, shifted_vpu, exp_block_vpu, 0U);
    templates->reduce_sum = primitiveBuildReduceExecCmd(
        config->device_id, VPU_OP_VREDUCE_SUM, exp_block_vpu, block_sum_vpu);
    templates->sub_prev = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VSUB, m_prev_vpu, m_next_vpu,
        prev_delta_vpu);
    templates->exp_prev = primitiveBuildUnaryExecCmd(
        config->device_id, VPU_OP_VEXP, prev_delta_vpu, prev_scale_vpu, 0U);
    templates->mul_prev_l = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VMUL, l_prev_vpu, prev_scale_vpu,
        prev_scaled_l_vpu);
    templates->add_l = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VADD, prev_scaled_l_vpu, block_sum_vpu,
        l_next_vpu);
    templates->div_prob = primitiveBuildBinaryExecCmd(
        config->device_id, VPU_OP_VDIV, exp_block_vpu, l_next_vpu,
        p_block_vpu);

    npu_fast_online_softmax_set_sync(&templates->reduce_max, config, false);
    npu_fast_online_softmax_set_sync(&templates->sub_max_delta, config, false);
    npu_fast_online_softmax_set_sync(&templates->abs_max_delta, config, false);
    npu_fast_online_softmax_set_sync(&templates->add_max_sum, config, false);
    npu_fast_online_softmax_set_sync(&templates->add_max_merge, config, false);
    npu_fast_online_softmax_set_sync(&templates->scale_m_next, config, false);
    npu_fast_online_softmax_set_sync(&templates->sub_scores, config, false);
    npu_fast_online_softmax_set_sync(&templates->exp_scores, config, false);
    npu_fast_online_softmax_set_sync(&templates->reduce_sum, config, false);
    npu_fast_online_softmax_set_sync(&templates->sub_prev, config, false);
    npu_fast_online_softmax_set_sync(&templates->exp_prev, config, false);
    npu_fast_online_softmax_set_sync(&templates->mul_prev_l, config, false);
    npu_fast_online_softmax_set_sync(&templates->add_l, config, false);
    npu_fast_online_softmax_set_sync(&templates->div_prob, config, true);
}

static inline size_t
vpu_fast_online_softmax_f32(
    const PrimitiveTensorDesc &scores_block, const PrimitiveTensorDesc &m_prev,
    const PrimitiveTensorDesc &l_prev, const PrimitiveTensorDesc &m_next,
    const PrimitiveTensorDesc &l_next, const PrimitiveTensorDesc &p_block,
    uint32_t scratch_base_slot,
    const NpuFastOnlineSoftmaxLaunchConfig &config,
    NpuFastOnlineSoftmaxStats *stats = nullptr)
{
    const PrimitiveTensorDesc scores2d = primitivePeelTo2D(scores_block, 0U);
    uint32_t slot = scratch_base_slot;
    const PrimitiveTensorDesc row_max =
        primitivePackedReduceLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(row_max);
    const PrimitiveTensorDesc max_delta =
        primitivePackedReduceLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(max_delta);
    const PrimitiveTensorDesc max_abs =
        primitivePackedReduceLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(max_abs);
    const PrimitiveTensorDesc max_sum =
        primitivePackedReduceLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(max_sum);
    const PrimitiveTensorDesc shifted = primitivePackedLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(shifted);
    const PrimitiveTensorDesc exp_block = primitivePackedLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(exp_block);
    const PrimitiveTensorDesc block_sum =
        primitivePackedReduceLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(block_sum);
    const PrimitiveTensorDesc prev_delta =
        primitivePackedReduceLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(prev_delta);
    const PrimitiveTensorDesc prev_scale =
        primitivePackedReduceLike(slot, scores2d);
    slot += primitiveTensorSlotSpan(prev_scale);
    const PrimitiveTensorDesc prev_scaled_l =
        primitivePackedReduceLike(slot, scores2d);
    NpuFastOnlineSoftmaxCmdTemplates templates = {};
    const uint64_t port_base =
        config.port_base == 0U ? NPU_CMD_PORT_BASE : config.port_base;

    if (npu_fast_online_softmax_validate_launch_config(&config) != 0 ||
        npu_fast_online_softmax_validate_tensor_descs(scores_block, m_prev,
                                                      l_prev, m_next, l_next,
                                                      p_block,
                                                      config.vpu_dlen_bytes) !=
            0) {
        return 0U;
    }

    const PrimitiveVpu2DDesc scores_vpu = primitiveLowerLastAxis2D(scores2d);
    const PrimitiveVpu2DDesc m_prev_vpu =
        primitiveLowerLastAxis2D(primitivePeelTo2D(m_prev, 0U));
    const PrimitiveVpu2DDesc l_prev_vpu =
        primitiveLowerLastAxis2D(primitivePeelTo2D(l_prev, 0U));
    const PrimitiveVpu2DDesc m_next_vpu =
        primitiveLowerLastAxis2D(primitivePeelTo2D(m_next, 0U));
    const PrimitiveVpu2DDesc l_next_vpu =
        primitiveLowerLastAxis2D(primitivePeelTo2D(l_next, 0U));
    const PrimitiveVpu2DDesc p_block_vpu =
        primitiveLowerLastAxis2D(primitivePeelTo2D(p_block, 0U));
    const PrimitiveVpu2DDesc row_max_vpu = primitiveLowerLastAxis2D(row_max);
    const PrimitiveVpu2DDesc max_delta_vpu =
        primitiveLowerLastAxis2D(max_delta);
    const PrimitiveVpu2DDesc max_abs_vpu = primitiveLowerLastAxis2D(max_abs);
    const PrimitiveVpu2DDesc max_sum_vpu = primitiveLowerLastAxis2D(max_sum);
    const PrimitiveVpu2DDesc shifted_vpu = primitiveLowerLastAxis2D(shifted);
    const PrimitiveVpu2DDesc exp_block_vpu =
        primitiveLowerLastAxis2D(exp_block);
    const PrimitiveVpu2DDesc block_sum_vpu =
        primitiveLowerLastAxis2D(block_sum);
    const PrimitiveVpu2DDesc prev_delta_vpu =
        primitiveLowerLastAxis2D(prev_delta);
    const PrimitiveVpu2DDesc prev_scale_vpu =
        primitiveLowerLastAxis2D(prev_scale);
    const PrimitiveVpu2DDesc prev_scaled_l_vpu =
        primitiveLowerLastAxis2D(prev_scaled_l);

    npu_fast_online_softmax_build_cmd_templates(
        scores_vpu, m_prev_vpu, l_prev_vpu, row_max_vpu, max_delta_vpu,
        max_abs_vpu, max_sum_vpu, m_next_vpu, shifted_vpu, exp_block_vpu,
        block_sum_vpu, prev_delta_vpu, prev_scale_vpu, prev_scaled_l_vpu,
        l_next_vpu, p_block_vpu, &config, &templates);

    if (stats != NULL) {
        *stats = {};
        stats->template_build_count = NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT;
        stats->slice_count = 1U;
    }

    templates.reduce_max.launchCmdAt(port_base);
    templates.sub_max_delta.launchCmdAt(port_base);
    templates.abs_max_delta.launchCmdAt(port_base);
    templates.add_max_sum.launchCmdAt(port_base);
    templates.add_max_merge.launchCmdAt(port_base);
    templates.scale_m_next.launchCmdAt(port_base);
    templates.sub_scores.launchCmdAt(port_base);
    templates.exp_scores.launchCmdAt(port_base);
    templates.reduce_sum.launchCmdAt(port_base);
    templates.sub_prev.launchCmdAt(port_base);
    templates.exp_prev.launchCmdAt(port_base);
    templates.mul_prev_l.launchCmdAt(port_base);
    templates.add_l.launchCmdAt(port_base);
    templates.div_prob.launchCmdAt(port_base);

    if (stats != NULL) {
        stats->launched_cmd_count = NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT;
    }

    return NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT;
}

#endif
