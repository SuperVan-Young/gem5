/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "primitive/fastOnlineSoftmax.hh"
#include "primitive/llm_test_utils.hh"

namespace
{

static constexpr uint32_t Rows = 256U;
static constexpr uint32_t Cols = 1024U;
static constexpr uint32_t MatrixSlotSpan =
    (Rows * Cols * sizeof(uint32_t) + VPU_LOCAL_SLOT_STRIDE - 1U) /
    VPU_LOCAL_SLOT_STRIDE;
static constexpr uint32_t VectorSlotSpan =
    (Rows * sizeof(uint32_t) + VPU_LOCAL_SLOT_STRIDE - 1U) /
    VPU_LOCAL_SLOT_STRIDE;
static constexpr uint32_t ScratchSlotSpan =
    MatrixSlotSpan * NPU_FAST_ONLINE_SOFTMAX_MATRIX_SCRATCH_COUNT +
    VectorSlotSpan * NPU_FAST_ONLINE_SOFTMAX_VECTOR_SCRATCH_COUNT;
static constexpr uint32_t ScoresSlot = 0U;
static constexpr uint32_t MPrevSlot = ScoresSlot + MatrixSlotSpan;
static constexpr uint32_t LPrevSlot = MPrevSlot + VectorSlotSpan;
static constexpr uint32_t MNextSlot = LPrevSlot + VectorSlotSpan;
static constexpr uint32_t LNextSlot = MNextSlot + VectorSlotSpan;
static constexpr uint32_t PBlockSlot = LNextSlot + VectorSlotSpan;
static constexpr uint32_t ScratchBaseSlot = PBlockSlot + MatrixSlotSpan;
static constexpr uint32_t VpuDeviceId = 0U;
static constexpr uint32_t SyncIndicator = 0x91U;
static constexpr float MTolerance = 1.0e-4f;
static constexpr float LTolerance = 3.0e-2f;
static constexpr float PTolerance = 3.0e-2f;
static constexpr const char *ExpectedScenario =
    "flash_attention_fast_online_softmax_256x1024";

static void
storeVector(uint32_t slot, const uint32_t *src, uint32_t count)
{
    volatile uint32_t *dst = npu_spm_slot_word_ptr_default(slot);
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = src[idx];
    }
}

static void
loadVector(uint32_t slot, uint32_t *dst, uint32_t count)
{
    const volatile uint32_t *src = npu_spm_slot_word_ptr_default(slot);
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = src[idx];
    }
}

static void
buildInputs(std::vector<uint32_t> &scores_bits,
            std::vector<uint32_t> &m_prev_bits,
            std::vector<uint32_t> &l_prev_bits)
{
    for (uint32_t row = 0U; row < Rows; ++row) {
        float row_max = -1.0e30f;

        for (uint32_t col = 0U; col < Cols; ++col) {
            const int32_t code =
                static_cast<int32_t>((row * 17U + col * 13U + 19U) % 113U) -
                56;
            const float value = static_cast<float>(code) * 0.125f;
            scores_bits[row * Cols + col] = npu_float_to_bits(value);
            if (value > row_max) {
                row_max = value;
            }
        }

        const float prev_m = (row & 0x1U) == 0U ? row_max - 0.45f :
                                                   row_max + 0.65f;
        const float prev_l =
            0.75f + static_cast<float>((row * 7U + 3U) % 23U) * 0.125f;
        m_prev_bits[row] = npu_float_to_bits(prev_m);
        l_prev_bits[row] = npu_float_to_bits(prev_l);
    }
}

static void
referenceOnlineSoftmax(const std::vector<uint32_t> &scores_bits,
                       const std::vector<uint32_t> &m_prev_bits,
                       const std::vector<uint32_t> &l_prev_bits,
                       std::vector<uint32_t> &m_next_bits,
                       std::vector<uint32_t> &l_next_bits,
                       std::vector<uint32_t> &p_block_bits)
{
    for (uint32_t row = 0U; row < Rows; ++row) {
        const uint32_t row_offset = row * Cols;
        float block_max = npu_bits_to_float(scores_bits[row_offset]);

        for (uint32_t col = 1U; col < Cols; ++col) {
            const float value =
                npu_bits_to_float(scores_bits[row_offset + col]);
            if (value > block_max) {
                block_max = value;
            }
        }

        const float prev_m = npu_bits_to_float(m_prev_bits[row]);
        const float prev_l = npu_bits_to_float(l_prev_bits[row]);
        const float next_m = prev_m > block_max ? prev_m : block_max;
        float block_sum = 0.0f;

        for (uint32_t col = 0U; col < Cols; ++col) {
            const float score =
                npu_bits_to_float(scores_bits[row_offset + col]);
            const float exp_value = expf(score - next_m);
            p_block_bits[row_offset + col] = npu_float_to_bits(exp_value);
            block_sum += exp_value;
        }

        const float next_l = prev_l * expf(prev_m - next_m) + block_sum;
        for (uint32_t col = 0U; col < Cols; ++col) {
            const float prob =
                npu_bits_to_float(p_block_bits[row_offset + col]) / next_l;
            p_block_bits[row_offset + col] = npu_float_to_bits(prob);
        }

        m_next_bits[row] = npu_float_to_bits(next_m);
        l_next_bits[row] = npu_float_to_bits(next_l);
    }
}

static float
maxAbsDiff(const std::vector<uint32_t> &expected,
           const std::vector<uint32_t> &actual)
{
    float max_diff = 0.0f;
    for (size_t idx = 0U; idx < expected.size(); ++idx) {
        const float diff =
            fabsf(npu_bits_to_float(expected[idx]) -
                  npu_bits_to_float(actual[idx]));
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

static float
sampleChecksum(const std::vector<uint32_t> &values)
{
    float checksum = 0.0f;
    for (uint32_t sample = 0U; sample < 32U; ++sample) {
        const uint32_t row = (sample * 19U) % Rows;
        const uint32_t col = (sample * 73U) % Cols;
        checksum += npu_bits_to_float(values[row * Cols + col]);
    }
    return checksum;
}

} // namespace

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : ExpectedScenario;
    const PrimitiveTensorDesc scores_desc =
        llm_packed_last_axis_tensor(ScoresSlot, Rows, Cols);
    const PrimitiveTensorDesc m_prev_desc =
        PrimitiveTensorDesc::denseSpm(MPrevSlot, VPU_DATA_F32, {Rows});
    const PrimitiveTensorDesc l_prev_desc =
        PrimitiveTensorDesc::denseSpm(LPrevSlot, VPU_DATA_F32, {Rows});
    const PrimitiveTensorDesc m_next_desc =
        PrimitiveTensorDesc::denseSpm(MNextSlot, VPU_DATA_F32, {Rows});
    const PrimitiveTensorDesc l_next_desc =
        PrimitiveTensorDesc::denseSpm(LNextSlot, VPU_DATA_F32, {Rows});
    const PrimitiveTensorDesc p_block_desc =
        llm_packed_last_axis_tensor(PBlockSlot, Rows, Cols);
    const NpuFastOnlineSoftmaxLaunchConfig config = {
        VpuDeviceId,
        SyncIndicator,
        1U,
        0U,
    };
    NpuFastOnlineSoftmaxStats stats = {};
    std::vector<uint32_t> scores_bits(Rows * Cols);
    std::vector<uint32_t> m_prev_bits(Rows);
    std::vector<uint32_t> l_prev_bits(Rows);
    std::vector<uint32_t> expected_m_next(Rows);
    std::vector<uint32_t> expected_l_next(Rows);
    std::vector<uint32_t> expected_p_block(Rows * Cols);
    std::vector<uint32_t> actual_m_next(Rows);
    std::vector<uint32_t> actual_l_next(Rows);
    std::vector<uint32_t> actual_p_block(Rows * Cols);

    if (strcmp(scenario, ExpectedScenario) != 0) {
        printf("FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_UNKNOWN_SCENARIO=%s\n",
               scenario);
        return 1;
    }
    if (ScratchSlotSpan !=
        npu_fast_online_softmax_scratch_slot_span(scores_desc)) {
        printf("FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SCRATCH_LAYOUT_FAIL\n");
        return 1;
    }

    buildInputs(scores_bits, m_prev_bits, l_prev_bits);
    referenceOnlineSoftmax(scores_bits, m_prev_bits, l_prev_bits,
                           expected_m_next, expected_l_next, expected_p_block);

    llm_clear_slot_span(ScoresSlot, MatrixSlotSpan);
    llm_clear_slot_span(MPrevSlot, VectorSlotSpan);
    llm_clear_slot_span(LPrevSlot, VectorSlotSpan);
    llm_clear_slot_span(MNextSlot, VectorSlotSpan);
    llm_clear_slot_span(LNextSlot, VectorSlotSpan);
    llm_clear_slot_span(PBlockSlot, MatrixSlotSpan);
    llm_clear_slot_span(ScratchBaseSlot, ScratchSlotSpan);

    llm_store_logical_matrix_last_axis_front(
        ScoresSlot, scores_bits.data(), Rows, Cols);
    storeVector(MPrevSlot, m_prev_bits.data(), Rows);
    storeVector(LPrevSlot, l_prev_bits.data(), Rows);

    const size_t macro_count = vpu_fast_online_softmax_f32(
        scores_desc, m_prev_desc, l_prev_desc, m_next_desc, l_next_desc,
        p_block_desc, ScratchBaseSlot, config, &stats);
    if (macro_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT ||
        stats.template_build_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT ||
        stats.launched_cmd_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT) {
        printf("FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CMD_FAIL "
               "macro_count=%u template_builds=%u launched=%u\n",
               (unsigned)macro_count, stats.template_build_count,
               stats.launched_cmd_count);
        return 1;
    }

    npu_launch_sync_wait(VpuDeviceId, SyncIndicator, 0U, 0U, 0U);
    npu_cmd_sync_done();

    loadVector(MNextSlot, actual_m_next.data(), Rows);
    loadVector(LNextSlot, actual_l_next.data(), Rows);
    llm_load_logical_matrix_last_axis_front(PBlockSlot, actual_p_block.data(),
                                            Rows, Cols);

    if (npu_expect_float_vector_close("FAST_ONLINE_SOFTMAX_M_NEXT",
                                      expected_m_next.data(),
                                      actual_m_next.data(), Rows, MTolerance,
                                      MTolerance) != 0 ||
        npu_expect_float_vector_close("FAST_ONLINE_SOFTMAX_L_NEXT",
                                      expected_l_next.data(),
                                      actual_l_next.data(), Rows, LTolerance,
                                      LTolerance) != 0 ||
        npu_expect_float_vector_close("FAST_ONLINE_SOFTMAX_P_BLOCK",
                                      expected_p_block.data(),
                                      actual_p_block.data(), Rows * Cols,
                                      PTolerance, PTolerance) != 0) {
        printf("FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_ACCURACY_FAIL\n");
        return 1;
    }

    printf("FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SCENARIO=%s\n", scenario);
    printf(
        "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_SHAPE rows=%u cols=%u "
        "total_elements=%u scratch_slots=%u\n",
        Rows, Cols, Rows * Cols, ScratchSlotSpan);
    printf(
        "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_CMD templates=%u launched=%u "
        "slices=%u sync_indicator=%u status=PASS\n",
        stats.template_build_count, stats.launched_cmd_count,
        stats.slice_count, SyncIndicator);
    printf(
        "FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_ACCURACY "
        "max_abs_m=%.8f max_abs_l=%.8f max_abs_p=%.8f sample_checksum=%.8f "
        "status=PASS\n",
        maxAbsDiff(expected_m_next, actual_m_next),
        maxAbsDiff(expected_l_next, actual_l_next),
        maxAbsDiff(expected_p_block, actual_p_block),
        sampleChecksum(actual_p_block));
    printf("FLASH_ATTENTION_FAST_ONLINE_SOFTMAX_PASS\n");
    return 0;
}
