/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "primitive/fastOnlineSoftmax.hh"
#include "primitive/llm_test_utils.hh"

namespace
{

constexpr uint32_t VpuDeviceId = 0U;
constexpr uint32_t SyncIndicator = 0x91U;
constexpr float MTolerance = 1.0e-4f;
constexpr float LTolerance = 3.0e-2f;
constexpr float PTolerance = 3.0e-2f;

uint32_t
parseU32(const char *text, const char *name)
{
    char *end = nullptr;
    const unsigned long value = strtoul(text, &end, 0);
    if (text[0] == '\0' || *end != '\0' || value == 0U ||
        value > UINT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return static_cast<uint32_t>(value);
}

void
storeVector(uint32_t slot, const uint32_t *src, uint32_t count)
{
    volatile uint32_t *dst = npu_spm_slot_word_ptr_default(slot);
    for (uint32_t index = 0U; index < count; ++index) {
        dst[index] = src[index];
    }
}

void
loadVector(uint32_t slot, uint32_t *dst, uint32_t count)
{
    const volatile uint32_t *src = npu_spm_slot_word_ptr_default(slot);
    for (uint32_t index = 0U; index < count; ++index) {
        dst[index] = src[index];
    }
}

float
scoreValue(uint32_t row, uint32_t col)
{
    const int32_t code =
        static_cast<int32_t>((row * 17U + col * 13U + 19U) % 113U) - 56;
    return static_cast<float>(code) * 0.125f;
}

void
buildInputs(uint32_t rows, uint32_t cols, std::vector<uint32_t> &scores,
            std::vector<uint32_t> &m_prev, std::vector<uint32_t> &l_prev)
{
    for (uint32_t row = 0U; row < rows; ++row) {
        float row_max = -1.0e30f;
        for (uint32_t col = 0U; col < cols; ++col) {
            const float value = scoreValue(row, col);
            scores[static_cast<size_t>(row) * cols + col] =
                npu_float_to_bits(value);
            row_max = value > row_max ? value : row_max;
        }
        const float previous_max =
            (row & 1U) == 0U ? row_max - 0.45f : row_max + 0.65f;
        const float previous_sum =
            0.75f + static_cast<float>((row * 7U + 3U) % 23U) * 0.125f;
        m_prev[row] = npu_float_to_bits(previous_max);
        l_prev[row] = npu_float_to_bits(previous_sum);
    }
}

bool
verifySamples(uint32_t rows, uint32_t cols,
              const std::vector<uint32_t> &m_prev,
              const std::vector<uint32_t> &l_prev,
              const std::vector<uint32_t> &m_next,
              const std::vector<uint32_t> &l_next,
              const std::vector<uint32_t> &probabilities)
{
    for (uint32_t sample = 0U; sample < 16U; ++sample) {
        const uint32_t row = (sample * 19U) % rows;
        const uint32_t col = (sample * 73U) % cols;
        float block_max = scoreValue(row, 0U);
        for (uint32_t index = 1U; index < cols; ++index) {
            const float value = scoreValue(row, index);
            block_max = value > block_max ? value : block_max;
        }
        const float previous_max = npu_bits_to_float(m_prev[row]);
        const float previous_sum = npu_bits_to_float(l_prev[row]);
        const float next_max =
            previous_max > block_max ? previous_max : block_max;
        float block_sum = 0.0f;
        for (uint32_t index = 0U; index < cols; ++index) {
            block_sum += expf(scoreValue(row, index) - next_max);
        }
        const float next_sum =
            previous_sum * expf(previous_max - next_max) + block_sum;
        const float probability =
            expf(scoreValue(row, col) - next_max) / next_sum;

        if (fabsf(npu_bits_to_float(m_next[row]) - next_max) > MTolerance ||
            fabsf(npu_bits_to_float(l_next[row]) - next_sum) > LTolerance ||
            fabsf(npu_bits_to_float(
                       probabilities[static_cast<size_t>(row) * cols + col]) -
                  probability) > PTolerance) {
            fprintf(stderr, "ATLAS_ONLINE_SOFTMAX_MISMATCH row=%u col=%u\n",
                    row, col);
            return false;
        }
    }
    return true;
}

} // namespace

int
main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s ROWS COLS DLEN_BYTES SPM_SIZE\n", argv[0]);
        return 2;
    }
    const uint32_t rows = parseU32(argv[1], "ROWS");
    const uint32_t cols = parseU32(argv[2], "COLS");
    const uint32_t dlen_bytes = parseU32(argv[3], "DLEN_BYTES");
    const uint32_t spm_size = parseU32(argv[4], "SPM_SIZE");
    const uint32_t layout_elems = dlen_bytes / sizeof(uint32_t);
    if (dlen_bytes % sizeof(uint32_t) != 0U ||
        rows % layout_elems != 0U || cols % layout_elems != 0U) {
        fprintf(stderr, "ATLAS_ONLINE_SOFTMAX_UNSUPPORTED_LAYOUT\n");
        return 2;
    }

    const uint32_t matrix_slots = llm_matrix_slot_span(rows, cols);
    const uint32_t vector_slots = llm_vector_slot_span(rows);
    const uint32_t scores_slot = 0U;
    const uint32_t m_prev_slot = scores_slot + matrix_slots;
    const uint32_t l_prev_slot = m_prev_slot + vector_slots;
    const uint32_t m_next_slot = l_prev_slot + vector_slots;
    const uint32_t l_next_slot = m_next_slot + vector_slots;
    const uint32_t p_block_slot = l_next_slot + vector_slots;
    const uint32_t scratch_base_slot = p_block_slot + matrix_slots;

    const PrimitiveTensorDesc scores_desc = llm_packed_last_axis_tensor(
        scores_slot, rows, cols, layout_elems);
    const PrimitiveTensorDesc m_prev_desc = PrimitiveTensorDesc::denseSpm(
        m_prev_slot, VPU_DATA_F32, {rows}, layout_elems);
    const PrimitiveTensorDesc l_prev_desc = PrimitiveTensorDesc::denseSpm(
        l_prev_slot, VPU_DATA_F32, {rows}, layout_elems);
    const PrimitiveTensorDesc m_next_desc = PrimitiveTensorDesc::denseSpm(
        m_next_slot, VPU_DATA_F32, {rows}, layout_elems);
    const PrimitiveTensorDesc l_next_desc = PrimitiveTensorDesc::denseSpm(
        l_next_slot, VPU_DATA_F32, {rows}, layout_elems);
    const PrimitiveTensorDesc p_block_desc = llm_packed_last_axis_tensor(
        p_block_slot, rows, cols, layout_elems);
    const uint32_t scratch_slots =
        npu_fast_online_softmax_scratch_slot_span(scores_desc);
    const uint64_t workspace_bytes =
        static_cast<uint64_t>(scratch_base_slot + scratch_slots) *
        VPU_LOCAL_SLOT_STRIDE;
    if (workspace_bytes > spm_size) {
        fprintf(stderr,
                "ATLAS_ONLINE_SOFTMAX_WORKSPACE_TOO_SMALL required=%llu "
                "available=%u\n",
                static_cast<unsigned long long>(workspace_bytes), spm_size);
        return 2;
    }

    std::vector<uint32_t> scores(static_cast<size_t>(rows) * cols);
    std::vector<uint32_t> m_prev(rows);
    std::vector<uint32_t> l_prev(rows);
    std::vector<uint32_t> m_next(rows);
    std::vector<uint32_t> l_next(rows);
    std::vector<uint32_t> probabilities(static_cast<size_t>(rows) * cols);
    buildInputs(rows, cols, scores, m_prev, l_prev);

    llm_clear_slot_span(scores_slot, matrix_slots);
    llm_clear_slot_span(m_prev_slot, vector_slots);
    llm_clear_slot_span(l_prev_slot, vector_slots);
    llm_clear_slot_span(m_next_slot, vector_slots);
    llm_clear_slot_span(l_next_slot, vector_slots);
    llm_clear_slot_span(p_block_slot, matrix_slots);
    llm_clear_slot_span(scratch_base_slot, scratch_slots);
    llm_store_logical_matrix_last_axis_front(scores_slot, scores.data(), rows,
                                             cols);
    storeVector(m_prev_slot, m_prev.data(), rows);
    storeVector(l_prev_slot, l_prev.data(), rows);

    const NpuFastOnlineSoftmaxLaunchConfig config = {
        VpuDeviceId, SyncIndicator, 1U, dlen_bytes, 0U};
    NpuFastOnlineSoftmaxStats stats = {};
    const size_t macro_count = vpu_fast_online_softmax_f32(
        scores_desc, m_prev_desc, l_prev_desc, m_next_desc, l_next_desc,
        p_block_desc, scratch_base_slot, config, &stats);
    if (macro_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT) {
        fprintf(stderr, "ATLAS_ONLINE_SOFTMAX_LAUNCH_FAIL\n");
        return 1;
    }
    npu_launch_sync_wait(VpuDeviceId, SyncIndicator, 0U, 0U, 0U);
    npu_cmd_sync_done();

    loadVector(m_next_slot, m_next.data(), rows);
    loadVector(l_next_slot, l_next.data(), rows);
    llm_load_logical_matrix_last_axis_front(
        p_block_slot, probabilities.data(), rows, cols);
    if (!verifySamples(rows, cols, m_prev, l_prev, m_next, l_next,
                       probabilities)) {
        return 1;
    }

    printf("ATLAS_ONLINE_SOFTMAX_WORKLOAD_PASS rows=%u cols=%u macros=%u "
           "slices=%u\n",
           rows, cols, stats.launched_cmd_count, stats.slice_count);
    return 0;
}
