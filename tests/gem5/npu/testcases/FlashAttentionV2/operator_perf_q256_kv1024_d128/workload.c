/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <climits>
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

static constexpr uint32_t M = 256U;
static constexpr uint32_t N = 1024U;
static constexpr uint32_t K = 128U;
static constexpr uint32_t TileM = 128U;
static constexpr uint32_t TileN = 128U;
static constexpr uint32_t TileK = K;
static constexpr uint32_t DeviceId = 0U;
static constexpr uint32_t QKSyncIndicator = 0x81U;
static constexpr uint32_t SoftmaxSyncIndicator = 0x91U;
static constexpr uint32_t PVSyncIndicator = 0x82U;
static constexpr uint64_t WaitTimeout = 80000000ULL;
static constexpr int32_t OutputSentinel = INT32_MIN;
static constexpr uintptr_t ASpm = 0x60000000ULL;
static constexpr uintptr_t BSpm = 0x60008000ULL;
static constexpr uintptr_t QKCSpm = 0x60130000ULL;
static constexpr uintptr_t PVCSpm = 0x60230000ULL;
static constexpr uint32_t LayoutSizeElems = 128U;
static constexpr uint32_t MatrixSlotSpan =
    (M * N * sizeof(uint32_t) + VPU_LOCAL_SLOT_STRIDE - 1U) /
    VPU_LOCAL_SLOT_STRIDE;
static constexpr uint32_t VectorSlotSpan =
    (M * sizeof(uint32_t) + VPU_LOCAL_SLOT_STRIDE - 1U) /
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
static constexpr const char *ExpectedScenario =
    "flash_attention_v2_operator_perf_q256_kv1024_d128";

static int8_t AHost[M * K];
static int8_t BHost[K * N];

static void
fillMatmulInputs(uint32_t seed)
{
    for (uint32_t row = 0U; row < M; ++row) {
        for (uint32_t col = 0U; col < K; ++col) {
            const int32_t code =
                (int32_t)((row * 17U + col * 13U + seed) % 31U) - 15;
            AHost[row * K + col] = (int8_t)code;
        }
    }

    for (uint32_t row = 0U; row < K; ++row) {
        for (uint32_t col = 0U; col < N; ++col) {
            const int32_t code =
                (int32_t)((row * 11U + col * 7U + seed) % 29U) - 14;
            BHost[row * N + col] = (int8_t)code;
        }
    }
}

static int
waitOutputValueReady(const NpuMpuGemmSpmI32Matrix *matrix, uint32_t row,
                     uint32_t col)
{
    const uintptr_t addr = npu_mpu_gemm_i32_addr(matrix, row, col);
    const volatile int32_t *value_ptr = (const volatile int32_t *)addr;

    return npu_wait_i32_vector_not_value(NULL, value_ptr, OutputSentinel, 1U,
                                         WaitTimeout);
}

static int
runFastMatmulPhase(const NpuMpuGemmSpmI8Matrix *a_matrix,
                   const NpuMpuGemmSpmI8Matrix *b_matrix,
                   const NpuMpuGemmSpmI32Matrix *c_matrix,
                   const NpuMpuGemmProblemShape *problem,
                   const NpuMpuGemmTileShape *tile,
                   uint32_t sync_indicator, uint32_t seed,
                   NpuFastMatmulStats *stats)
{
    const NpuFastMatmulLaunchConfig launch = {
        DeviceId, sync_indicator, 0U, 0U, 0U, 0U, 0U,
    };

    fillMatmulInputs(seed);
    npu_mpu_gemm_store_i8_matrix_to_spm(AHost, a_matrix);
    npu_mpu_gemm_store_i8_matrix_to_spm(BHost, b_matrix);
    npu_mpu_gemm_fill_i32_matrix_in_spm(c_matrix, OutputSentinel);

    if (npu_fast_matmul_launch_tiled_spm(a_matrix, b_matrix, c_matrix,
                                         problem, tile, &launch,
                                         stats) != 0) {
        return -1;
    }
    if (waitOutputValueReady(c_matrix, c_matrix->rows - 1U,
                             c_matrix->cols - 1U) != 0) {
        return -1;
    }
    return 0;
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
buildSoftmaxInputs(std::vector<uint32_t> &scores_bits,
                   std::vector<uint32_t> &m_prev_bits,
                   std::vector<uint32_t> &l_prev_bits)
{
    for (uint32_t row = 0U; row < M; ++row) {
        float row_max = -1.0e30f;

        for (uint32_t col = 0U; col < N; ++col) {
            const int32_t code =
                (int32_t)((row * 17U + col * 13U + 19U) % 113U) - 56;
            const float value = (float)code * 0.125f;
            scores_bits[row * N + col] = npu_float_to_bits(value);
            if (value > row_max) {
                row_max = value;
            }
        }

        m_prev_bits[row] = npu_float_to_bits(row_max - 0.45f);
        l_prev_bits[row] = npu_float_to_bits(
            0.75f + (float)((row * 7U + 3U) % 23U) * 0.125f);
    }
}

static int
runFastOnlineSoftmaxPhase()
{
    const PrimitiveTensorDesc scores_desc =
        llm_packed_last_axis_tensor(ScoresSlot, M, N, LayoutSizeElems);
    const PrimitiveTensorDesc m_prev_desc =
        PrimitiveTensorDesc::denseSpm(
            MPrevSlot, VPU_DATA_F32, {M}, LayoutSizeElems);
    const PrimitiveTensorDesc l_prev_desc =
        PrimitiveTensorDesc::denseSpm(
            LPrevSlot, VPU_DATA_F32, {M}, LayoutSizeElems);
    const PrimitiveTensorDesc m_next_desc =
        PrimitiveTensorDesc::denseSpm(
            MNextSlot, VPU_DATA_F32, {M}, LayoutSizeElems);
    const PrimitiveTensorDesc l_next_desc =
        PrimitiveTensorDesc::denseSpm(
            LNextSlot, VPU_DATA_F32, {M}, LayoutSizeElems);
    const PrimitiveTensorDesc p_block_desc =
        llm_packed_last_axis_tensor(PBlockSlot, M, N, LayoutSizeElems);
    const NpuFastOnlineSoftmaxLaunchConfig config = {
        DeviceId, SoftmaxSyncIndicator, 1U, 512U, 0U,
    };
    NpuFastOnlineSoftmaxStats stats = {};
    std::vector<uint32_t> scores_bits(M * N);
    std::vector<uint32_t> m_prev_bits(M);
    std::vector<uint32_t> l_prev_bits(M);

    if (ScratchSlotSpan !=
        npu_fast_online_softmax_scratch_slot_span(scores_desc)) {
        printf("FLASH_ATTENTION_V2_SOFTMAX_SCRATCH_LAYOUT_FAIL\n");
        return -1;
    }

    buildSoftmaxInputs(scores_bits, m_prev_bits, l_prev_bits);
    llm_clear_slot_span(ScoresSlot, MatrixSlotSpan);
    llm_clear_slot_span(MPrevSlot, VectorSlotSpan);
    llm_clear_slot_span(LPrevSlot, VectorSlotSpan);
    llm_clear_slot_span(MNextSlot, VectorSlotSpan);
    llm_clear_slot_span(LNextSlot, VectorSlotSpan);
    llm_clear_slot_span(PBlockSlot, MatrixSlotSpan);
    llm_clear_slot_span(ScratchBaseSlot, ScratchSlotSpan);
    llm_store_logical_matrix_last_axis_front(
        ScoresSlot, scores_bits.data(), M, N);
    storeVector(MPrevSlot, m_prev_bits.data(), M);
    storeVector(LPrevSlot, l_prev_bits.data(), M);

    const size_t macro_count = vpu_fast_online_softmax_f32(
        scores_desc, m_prev_desc, l_prev_desc, m_next_desc, l_next_desc,
        p_block_desc, ScratchBaseSlot, config, &stats);
    if (macro_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT ||
        stats.launched_cmd_count != NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT) {
        printf("FLASH_ATTENTION_V2_SOFTMAX_CMD_FAIL macro_count=%u "
               "launched=%u\n",
               (unsigned)macro_count, stats.launched_cmd_count);
        return -1;
    }

    npu_launch_sync_wait(DeviceId, SoftmaxSyncIndicator, 0U, 0U, 0U);
    npu_cmd_sync_done();
    return 0;
}

static uint64_t
singleMatmulFlops()
{
    return 2ULL * (uint64_t)M * (uint64_t)N * (uint64_t)K;
}

} // namespace

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : ExpectedScenario;
    const NpuMpuGemmSpmI8Matrix a_matrix =
        npu_mpu_gemm_i8_blocked_matrix(ASpm, M, K, TileM, TileK);
    const NpuMpuGemmSpmI8Matrix b_matrix =
        npu_mpu_gemm_i8_blocked_matrix(BSpm, K, N, TileK, TileN);
    const NpuMpuGemmSpmI32Matrix qk_c_matrix =
        npu_mpu_gemm_i32_blocked_matrix(QKCSpm, M, N, TileM, TileN);
    const NpuMpuGemmSpmI32Matrix pv_c_matrix =
        npu_mpu_gemm_i32_blocked_matrix(PVCSpm, M, N, TileM, TileN);
    const NpuMpuGemmProblemShape problem = {M, N, K};
    const NpuMpuGemmTileShape tile = {TileM, TileN, TileK};
    NpuFastMatmulStats qk_stats = {};
    NpuFastMatmulStats pv_stats = {};

    if (strcmp(scenario, ExpectedScenario) != 0) {
        printf("FLASH_ATTENTION_V2_UNKNOWN_SCENARIO=%s\n", scenario);
        return 1;
    }

    if (runFastMatmulPhase(&a_matrix, &b_matrix, &qk_c_matrix, &problem,
                           &tile, QKSyncIndicator, 0U, &qk_stats) != 0) {
        printf("FLASH_ATTENTION_V2_QK_FAST_MATMUL_FAIL\n");
        return 1;
    }
    if (runFastOnlineSoftmaxPhase() != 0) {
        printf("FLASH_ATTENTION_V2_FAST_ONLINE_SOFTMAX_FAIL\n");
        return 1;
    }
    if (runFastMatmulPhase(&a_matrix, &b_matrix, &pv_c_matrix, &problem,
                           &tile, PVSyncIndicator, 5U, &pv_stats) != 0) {
        printf("FLASH_ATTENTION_V2_PV_FAST_MATMUL_FAIL\n");
        return 1;
    }

    printf("FLASH_ATTENTION_V2_SCENARIO=%s\n", scenario);
    printf("FLASH_ATTENTION_V2_SHAPE q=%u kv=%u d=%u tile_m=%u tile_n=%u "
           "tile_k=%u matmul_count=2 single_matmul_flops=%llu "
           "total_matmul_flops=%llu\n",
           M, N, K, TileM, TileN, TileK,
           (unsigned long long)singleMatmulFlops(),
           (unsigned long long)(singleMatmulFlops() * 2ULL));
    printf("FLASH_ATTENTION_V2_QK_FAST_MATMUL cmds=%u template_builds=%u "
           "sync_indicator=%u status=PASS\n",
           qk_stats.launched_cmd_count, qk_stats.template_build_count,
           QKSyncIndicator);
    printf("FLASH_ATTENTION_V2_FAST_ONLINE_SOFTMAX cmds=%u "
           "sync_indicator=%u status=PASS\n",
           NPU_FAST_ONLINE_SOFTMAX_TEMPLATE_COUNT, SoftmaxSyncIndicator);
    printf("FLASH_ATTENTION_V2_PV_FAST_MATMUL cmds=%u template_builds=%u "
           "sync_indicator=%u status=PASS\n",
           pv_stats.launched_cmd_count, pv_stats.template_build_count,
           PVSyncIndicator);
    printf("FLASH_ATTENTION_V2_PASS\n");
    return 0;
}
