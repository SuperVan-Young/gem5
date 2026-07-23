/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <vector>

#include "mpu_gemm.hh"
#include "npu_assert.hh"
#include "npu_mem.hh"
#include "npu_sync.hh"
#include "primitive/fastMatmul.hh"
#include "primitive/fastOnlineSoftmax.hh"
#include "primitive/llm_test_utils.hh"

namespace
{

static constexpr uint32_t ArrayDim = 128U;
static constexpr uint32_t DeviceId = 0U;
static constexpr uint32_t QKSyncIndicator = 0x81U;
static constexpr uint32_t SoftmaxSyncIndicator = 0x91U;
static constexpr uint32_t PVSyncIndicator = 0x82U;
static constexpr uint32_t LayoutSizeElems = 128U;
static constexpr uint32_t VpuDlenBytes = 512U;
static constexpr uint64_t WaitTimeout = 80000000ULL;
static constexpr int32_t OutputSentinel = INT32_MIN;
static constexpr uintptr_t SpmBase = 0x60000000ULL;
static constexpr uint64_t SpmSize = 8ULL * 1024ULL * 1024ULL;
static constexpr const char *ExpectedScenario =
    "flash_attention_v2_operator_perf_q256_kv1024_d128";

struct TaskShape
{
    uint32_t q = 256U;
    uint32_t kv = 1024U;
    uint32_t d = 128U;
    const char *scenario = ExpectedScenario;
};

static bool
parseU32(const char *text, uint32_t *value)
{
    char *end = nullptr;
    unsigned long parsed = 0;

    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0UL ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

static bool
parseArgs(int argc, char **argv, TaskShape *shape)
{
    for (int index = 1; index < argc; ++index) {
        const char *arg = argv[index];
        uint32_t *target = nullptr;
        if (strcmp(arg, "--q") == 0) {
            target = &shape->q;
        } else if (strcmp(arg, "--kv") == 0) {
            target = &shape->kv;
        } else if (strcmp(arg, "--d") == 0) {
            target = &shape->d;
        } else if (strcmp(arg, "--scenario") == 0) {
            if (++index >= argc) {
                return false;
            }
            shape->scenario = argv[index];
            continue;
        } else if (arg[0] != '-' && index == 1) {
            shape->scenario = arg;
            continue;
        } else {
            return false;
        }
        if (++index >= argc || !parseU32(argv[index], target)) {
            return false;
        }
    }

    return strcmp(shape->scenario, ExpectedScenario) == 0 &&
           shape->d <= 255U;
}

static bool
checkedElements(uint32_t rows, uint32_t cols, size_t *elements)
{
    const uint64_t count = static_cast<uint64_t>(rows) * cols;
    if (count == 0U ||
        count > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
        return false;
    }
    *elements = static_cast<size_t>(count);
    return true;
}

static uint64_t
alignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1U) / alignment) * alignment;
}

static uint64_t
blockedBytes(uint32_t rows, uint32_t cols, uint32_t block_rows,
             uint32_t block_cols, uint32_t elem_bytes)
{
    const uint64_t row_blocks = (rows + block_rows - 1U) / block_rows;
    const uint64_t col_blocks = (cols + block_cols - 1U) / block_cols;
    return row_blocks * col_blocks * block_rows * block_cols * elem_bytes;
}

struct MatmulWorkspace
{
    uintptr_t a;
    uintptr_t b;
    uintptr_t c;
    uint64_t end;
};

static bool
makeWorkspace(const TaskShape &shape, MatmulWorkspace *workspace)
{
    const uint32_t q_tile_m = std::min(shape.q, ArrayDim);
    const uint32_t qk_tile_n = std::min(shape.kv, ArrayDim);
    const uint32_t pv_tile_n = std::min(shape.d, ArrayDim);
    const uint32_t pv_tile_k = std::min(shape.kv, ArrayDim);
    const uint64_t qk_a = blockedBytes(
        shape.q, shape.d, q_tile_m, shape.d, sizeof(int8_t));
    const uint64_t qk_b = blockedBytes(
        shape.d, shape.kv, shape.d, qk_tile_n, sizeof(int8_t));
    const uint64_t qk_c = blockedBytes(
        shape.q, shape.kv, q_tile_m, qk_tile_n, sizeof(int32_t));
    const uint64_t pv_a = blockedBytes(
        shape.q, pv_tile_k, q_tile_m, pv_tile_k, sizeof(int8_t));
    const uint64_t pv_b = blockedBytes(
        pv_tile_k, shape.d, pv_tile_k, pv_tile_n, sizeof(int8_t));
    const uint64_t pv_c = blockedBytes(
        shape.q, shape.d, q_tile_m, pv_tile_n, sizeof(int32_t));
    const uint64_t a_bytes = std::max(qk_a, pv_a);
    const uint64_t b_bytes = std::max(qk_b, pv_b);
    const uint64_t c_bytes = std::max(qk_c, pv_c);

    workspace->a = SpmBase;
    workspace->b = alignUp(workspace->a + a_bytes, VPU_LOCAL_SLOT_STRIDE);
    workspace->c = alignUp(workspace->b + b_bytes, VPU_LOCAL_SLOT_STRIDE);
    workspace->end =
        alignUp(workspace->c + c_bytes, VPU_LOCAL_SLOT_STRIDE);
    return workspace->end <= SpmBase + SpmSize;
}

static void
fillMatmulInputs(std::vector<int8_t> &a, std::vector<int8_t> &b, uint32_t m,
                 uint32_t n, uint32_t k, uint32_t seed)
{
    for (uint32_t row = 0U; row < m; ++row) {
        for (uint32_t col = 0U; col < k; ++col) {
            const int32_t code =
                static_cast<int32_t>((row * 17U + col * 13U + seed) % 31U) -
                15;
            a[static_cast<size_t>(row) * k + col] =
                static_cast<int8_t>(code);
        }
    }
    for (uint32_t row = 0U; row < k; ++row) {
        for (uint32_t col = 0U; col < n; ++col) {
            const int32_t code =
                static_cast<int32_t>((row * 11U + col * 7U + seed) % 29U) -
                14;
            b[static_cast<size_t>(row) * n + col] =
                static_cast<int8_t>(code);
        }
    }
}

static int
waitOutputValueReady(const NpuMpuGemmSpmI32Matrix *matrix)
{
    const uintptr_t addr = npu_mpu_gemm_i32_addr(
        matrix, matrix->rows - 1U, matrix->cols - 1U);
    const volatile int32_t *value_ptr =
        reinterpret_cast<const volatile int32_t *>(addr);
    return npu_wait_i32_vector_not_value(
        nullptr, value_ptr, OutputSentinel, 1U, WaitTimeout);
}

static int
runFastMatmul(const MatmulWorkspace &workspace, uint32_t m, uint32_t n,
              uint32_t k, uint32_t sync_indicator, uint32_t seed,
              NpuFastMatmulStats *stats)
{
    const uint32_t tile_m = std::min(m, ArrayDim);
    const uint32_t tile_n = std::min(n, ArrayDim);
    const NpuMpuGemmProblemShape problem = {m, n, k};
    const NpuMpuGemmTileShape tile = {tile_m, tile_n, k};
    const NpuMpuGemmSpmI8Matrix a_matrix =
        npu_mpu_gemm_i8_blocked_matrix(workspace.a, m, k, tile_m, k);
    const NpuMpuGemmSpmI8Matrix b_matrix =
        npu_mpu_gemm_i8_blocked_matrix(workspace.b, k, n, k, tile_n);
    const NpuMpuGemmSpmI32Matrix c_matrix =
        npu_mpu_gemm_i32_blocked_matrix(
            workspace.c, m, n, tile_m, tile_n);
    const NpuFastMatmulLaunchConfig launch = {
        DeviceId, sync_indicator, 0U, 0U, 0U, 0U, 0U,
    };
    size_t a_elements = 0;
    size_t b_elements = 0;
    if (!checkedElements(m, k, &a_elements) ||
        !checkedElements(k, n, &b_elements)) {
        return -1;
    }
    std::vector<int8_t> a(a_elements);
    std::vector<int8_t> b(b_elements);

    fillMatmulInputs(a, b, m, n, k, seed);
    npu_mpu_gemm_store_i8_matrix_to_spm(a.data(), &a_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(b.data(), &b_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(&c_matrix, OutputSentinel);
    if (npu_fast_matmul_launch_tiled_spm(
            &a_matrix, &b_matrix, &c_matrix, &problem, &tile, &launch,
            stats) != 0) {
        return -1;
    }
    return waitOutputValueReady(&c_matrix);
}

static void
storeVector(uint32_t slot, const uint32_t *src, uint32_t count)
{
    volatile uint32_t *dst = npu_spm_slot_word_ptr_default(slot);
    for (uint32_t idx = 0U; idx < count; ++idx) {
        dst[idx] = src[idx];
    }
}

static void
buildSoftmaxInputs(const TaskShape &shape,
                   std::vector<uint32_t> &scores_bits,
                   std::vector<uint32_t> &m_prev_bits,
                   std::vector<uint32_t> &l_prev_bits)
{
    for (uint32_t row = 0U; row < shape.q; ++row) {
        float row_max = -1.0e30f;
        for (uint32_t col = 0U; col < shape.kv; ++col) {
            const int32_t code =
                static_cast<int32_t>(
                    (row * 17U + col * 13U + 19U) % 113U) -
                56;
            const float value = static_cast<float>(code) * 0.125f;
            scores_bits[static_cast<size_t>(row) * shape.kv + col] =
                npu_float_to_bits(value);
            row_max = std::max(row_max, value);
        }
        m_prev_bits[row] = npu_float_to_bits(row_max - 0.45f);
        l_prev_bits[row] = npu_float_to_bits(
            0.75f + static_cast<float>((row * 7U + 3U) % 23U) * 0.125f);
    }
}

static int
runFastOnlineSoftmaxPhase(const TaskShape &shape)
{
    const uint32_t matrix_slot_span =
        llm_matrix_slot_span(shape.q, shape.kv);
    const uint32_t vector_slot_span = llm_vector_slot_span(shape.q);
    const uint32_t scores_slot = 0U;
    const uint32_t m_prev_slot = scores_slot + matrix_slot_span;
    const uint32_t l_prev_slot = m_prev_slot + vector_slot_span;
    const uint32_t m_next_slot = l_prev_slot + vector_slot_span;
    const uint32_t l_next_slot = m_next_slot + vector_slot_span;
    const uint32_t p_block_slot = l_next_slot + vector_slot_span;
    const uint32_t scratch_base_slot = p_block_slot + matrix_slot_span;
    const PrimitiveTensorDesc scores_desc = llm_packed_last_axis_tensor(
        scores_slot, shape.q, shape.kv, LayoutSizeElems);
    const PrimitiveTensorDesc m_prev_desc = PrimitiveTensorDesc::denseSpm(
        m_prev_slot, VPU_DATA_F32, {shape.q}, LayoutSizeElems);
    const PrimitiveTensorDesc l_prev_desc = PrimitiveTensorDesc::denseSpm(
        l_prev_slot, VPU_DATA_F32, {shape.q}, LayoutSizeElems);
    const PrimitiveTensorDesc m_next_desc = PrimitiveTensorDesc::denseSpm(
        m_next_slot, VPU_DATA_F32, {shape.q}, LayoutSizeElems);
    const PrimitiveTensorDesc l_next_desc = PrimitiveTensorDesc::denseSpm(
        l_next_slot, VPU_DATA_F32, {shape.q}, LayoutSizeElems);
    const PrimitiveTensorDesc p_block_desc = llm_packed_last_axis_tensor(
        p_block_slot, shape.q, shape.kv, LayoutSizeElems);
    const uint32_t scratch_slot_span =
        npu_fast_online_softmax_scratch_slot_span(scores_desc);
    const uint64_t end_bytes =
        static_cast<uint64_t>(scratch_base_slot + scratch_slot_span) *
        VPU_LOCAL_SLOT_STRIDE;
    const NpuFastOnlineSoftmaxLaunchConfig config = {
        DeviceId, SoftmaxSyncIndicator, 1U, VpuDlenBytes, 0U,
    };
    NpuFastOnlineSoftmaxStats stats = {};
    size_t score_elements = 0;
    if (!checkedElements(shape.q, shape.kv, &score_elements) ||
        end_bytes > SpmSize) {
        printf("FLASH_ATTENTION_V2_SOFTMAX_SPM_CAPACITY_FAIL "
               "required=%llu available=%llu\n",
               static_cast<unsigned long long>(end_bytes),
               static_cast<unsigned long long>(SpmSize));
        return -1;
    }
    std::vector<uint32_t> scores_bits(score_elements);
    std::vector<uint32_t> m_prev_bits(shape.q);
    std::vector<uint32_t> l_prev_bits(shape.q);

    buildSoftmaxInputs(shape, scores_bits, m_prev_bits, l_prev_bits);
    llm_clear_slot_span(scores_slot, matrix_slot_span);
    llm_clear_slot_span(m_prev_slot, vector_slot_span);
    llm_clear_slot_span(l_prev_slot, vector_slot_span);
    llm_clear_slot_span(m_next_slot, vector_slot_span);
    llm_clear_slot_span(l_next_slot, vector_slot_span);
    llm_clear_slot_span(p_block_slot, matrix_slot_span);
    llm_clear_slot_span(scratch_base_slot, scratch_slot_span);
    llm_store_logical_matrix_last_axis_front(
        scores_slot, scores_bits.data(), shape.q, shape.kv);
    storeVector(m_prev_slot, m_prev_bits.data(), shape.q);
    storeVector(l_prev_slot, l_prev_bits.data(), shape.q);

    const size_t macro_count = vpu_fast_online_softmax_f32(
        scores_desc, m_prev_desc, l_prev_desc, m_next_desc, l_next_desc,
        p_block_desc, scratch_base_slot, config, &stats);
    if (macro_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT ||
        stats.launched_cmd_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT) {
        return -1;
    }
    npu_launch_sync_wait(DeviceId, SoftmaxSyncIndicator, 0U, 0U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static uint64_t
singleStageOps(const TaskShape &shape)
{
    return 2ULL * shape.q * shape.kv * shape.d;
}

} // namespace

int
main(int argc, char **argv)
{
    TaskShape shape;
    MatmulWorkspace workspace = {};
    NpuFastMatmulStats qk_stats = {};
    uint32_t pv_cmds = 0U;
    uint32_t pv_templates = 0U;

    if (!parseArgs(argc, argv, &shape)) {
        printf("FLASH_ATTENTION_V2_ARGUMENT_FAIL\n");
        return 1;
    }
    if (!makeWorkspace(shape, &workspace)) {
        printf("FLASH_ATTENTION_V2_SPM_CAPACITY_FAIL\n");
        return 1;
    }
    if (runFastMatmul(workspace, shape.q, shape.kv, shape.d,
                      QKSyncIndicator, 0U, &qk_stats) != 0) {
        printf("FLASH_ATTENTION_V2_QK_FAST_MATMUL_FAIL\n");
        return 1;
    }
    if (runFastOnlineSoftmaxPhase(shape) != 0) {
        printf("FLASH_ATTENTION_V2_FAST_ONLINE_SOFTMAX_FAIL\n");
        return 1;
    }
    for (uint32_t k0 = 0U; k0 < shape.kv; k0 += ArrayDim) {
        const uint32_t chunk_k = std::min(ArrayDim, shape.kv - k0);
        NpuFastMatmulStats chunk_stats = {};
        if (runFastMatmul(workspace, shape.q, shape.d, chunk_k,
                          PVSyncIndicator, 5U + k0, &chunk_stats) != 0) {
            printf("FLASH_ATTENTION_V2_PV_FAST_MATMUL_FAIL k0=%u\n", k0);
            return 1;
        }
        pv_cmds += chunk_stats.launched_cmd_count;
        pv_templates += chunk_stats.template_build_count;
    }

    const uint64_t stage_ops = singleStageOps(shape);
    printf("FLASH_ATTENTION_V2_SCENARIO=%s\n", shape.scenario);
    printf("FLASH_ATTENTION_V2_SHAPE q=%u kv=%u d=%u tile_m=%u tile_n=%u "
           "tile_k=%u matmul_count=2 single_matmul_flops=%llu "
           "total_matmul_flops=%llu\n",
           shape.q, shape.kv, shape.d, std::min(shape.q, ArrayDim),
           std::min(shape.kv, ArrayDim), shape.d,
           static_cast<unsigned long long>(stage_ops),
           static_cast<unsigned long long>(stage_ops * 2ULL));
    printf("FLASH_ATTENTION_V2_QK_FAST_MATMUL cmds=%u template_builds=%u "
           "sync_indicator=%u status=PASS\n",
           qk_stats.launched_cmd_count, qk_stats.template_build_count,
           QKSyncIndicator);
    printf("FLASH_ATTENTION_V2_FAST_ONLINE_SOFTMAX cmds=%u "
           "sync_indicator=%u status=PASS\n",
           NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT, SoftmaxSyncIndicator);
    printf("FLASH_ATTENTION_V2_PV_FAST_MATMUL cmds=%u template_builds=%u "
           "sync_indicator=%u status=PASS\n",
           pv_cmds, pv_templates, PVSyncIndicator);
    printf("FLASH_ATTENTION_V2_SPM_LAYOUT a=%#llx b=%#llx c=%#llx end=%#llx "
           "capacity_end=%#llx\n",
           static_cast<unsigned long long>(workspace.a),
           static_cast<unsigned long long>(workspace.b),
           static_cast<unsigned long long>(workspace.c),
           static_cast<unsigned long long>(workspace.end),
           static_cast<unsigned long long>(SpmBase + SpmSize));
    printf("FLASH_ATTENTION_V2_PASS\n");
    return 0;
}
