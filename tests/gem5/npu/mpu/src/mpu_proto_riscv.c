#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mpu.hh"

#define DMA_DEVICE_ID 0x0U
#define DMA_MODE_MOVE_LAYOUT 0x0U
#define DMA_MEM_SPACE_DRAM 0x0U
#define DMA_MEM_SPACE_SPM 0x1U
#define DMA_CUT_DIM_W 0x1U

#define SPM_BASE 0x60000000UL
#define DRAM_BASE 0x20000000UL
#define MAX_WAIT_ITERS 2000000UL

#define A0_SPM (SPM_BASE + 0x1000UL)
#define B0_SPM (SPM_BASE + 0x2000UL)
#define C0_SPM (SPM_BASE + 0x3000UL)
#define A1_SPM (SPM_BASE + 0x4000UL)
#define B1_SPM (SPM_BASE + 0x5000UL)
#define C1_SPM (SPM_BASE + 0x6000UL)

#define DRAM_A0 (DRAM_BASE + 0x1000UL)
#define DRAM_B0 (DRAM_BASE + 0x2000UL)

static inline uint32_t
u32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

static inline void
write_u8(uintptr_t addr, uint8_t value)
{
    *(volatile uint8_t *)addr = value;
}

static inline uint8_t
read_u8(uintptr_t addr)
{
    return *(volatile uint8_t *)addr;
}

static inline void
write_i32(uintptr_t addr, int32_t value)
{
    *(volatile int32_t *)addr = value;
}

static inline int32_t
read_i32(uintptr_t addr)
{
    return *(volatile int32_t *)addr;
}

static void
clear_bytes(uintptr_t base, size_t bytes)
{
    for (size_t i = 0; i < bytes; ++i) {
        write_u8(base + i, 0U);
    }
}

static void
fill_matrix_i8(uintptr_t base, uint32_t rows, uint32_t cols,
               uint32_t stride_bytes, const int8_t *values)
{
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            write_u8(base + row * stride_bytes + col,
                     (uint8_t)values[row * cols + col]);
        }
    }
}

static void
fill_matrix_i32(uintptr_t base, uint32_t rows, uint32_t cols,
                uint32_t stride_bytes, int32_t value)
{
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            write_i32(base + row * stride_bytes + col * sizeof(int32_t),
                      value);
        }
    }
}

static int
matrix_matches_i8(uintptr_t base, uint32_t rows, uint32_t cols,
                  uint32_t stride_bytes, const int8_t *expected)
{
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            if ((int8_t)read_u8(base + row * stride_bytes + col) !=
                expected[row * cols + col]) {
                return 0;
            }
        }
    }
    return 1;
}

static int
matrix_matches_i32(uintptr_t base, uint32_t rows, uint32_t cols,
                   uint32_t stride_bytes, const int32_t *expected)
{
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            if (read_i32(base + row * stride_bytes + col * sizeof(int32_t)) !=
                expected[row * cols + col]) {
                return 0;
            }
        }
    }
    return 1;
}

static int
spin_until_i8_match(uintptr_t base, uint32_t rows, uint32_t cols,
                    uint32_t stride_bytes, const int8_t *expected)
{
    for (unsigned long iter = 0; iter < MAX_WAIT_ITERS; ++iter) {
        if (matrix_matches_i8(base, rows, cols, stride_bytes, expected)) {
            return 1;
        }
    }
    return 0;
}

static int
spin_until_i32_match(uintptr_t base, uint32_t rows, uint32_t cols,
                     uint32_t stride_bytes, const int32_t *expected)
{
    for (unsigned long iter = 0; iter < MAX_WAIT_ITERS; ++iter) {
        if (matrix_matches_i32(base, rows, cols, stride_bytes, expected)) {
            return 1;
        }
    }
    return 0;
}

static void
compute_expected(const int8_t *a, const int8_t *b,
                 uint32_t m, uint32_t n, uint32_t k,
                 int32_t *out)
{
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            int32_t acc = 0;
            for (uint32_t depth = 0; depth < k; ++depth) {
                acc += (int32_t)a[row * k + depth] *
                       (int32_t)b[depth * n + col];
            }
            out[row * n + col] = acc;
        }
    }
}

static inline void
launch_mpu_mvin(uint32_t device_id, uint32_t buffer_kind,
                uint32_t buffer_index, uint32_t m, uint32_t n, uint32_t k,
                uintptr_t spm_addr, uint32_t stride_bytes)
{
    mpu_launch_cmd(device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVIN),
                   buffer_kind, buffer_index,
                   m, n, k, spm_addr, stride_bytes, 0U, 0U);
}

static inline void
launch_mpu_load(uint32_t device_id, uint32_t buffer_kind,
                uint32_t buffer_index, uint32_t m, uint32_t n, uint32_t k)
{
    mpu_launch_cmd(device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_LOAD),
                   buffer_kind, buffer_index,
                   m, n, k, 0U, 0U, 0U, 0U);
}

static inline void
launch_mpu_compute(uint32_t device_id, uint32_t m, uint32_t n, uint32_t k)
{
    mpu_launch_cmd(device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_COMPUTE),
                   MPU_BUFFER_RESERVED, 0U,
                   m, n, k, 0U, 0U, 0U, 0U);
}

static inline void
launch_mpu_drain(uint32_t device_id, uint32_t buffer_index,
                 uint32_t m, uint32_t n, uint32_t k)
{
    mpu_launch_cmd(device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_DRAIN),
                   MPU_BUFFER_C, buffer_index,
                   m, n, k, 0U, 0U, 0U, 0U);
}

static inline void
launch_mpu_mvout(uint32_t device_id, uint32_t buffer_index,
                 uint32_t m, uint32_t n, uint32_t k,
                 uintptr_t spm_addr, uint32_t stride_bytes)
{
    mpu_launch_cmd(device_id,
                   mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVOUT),
                   MPU_BUFFER_C, buffer_index,
                   m, n, k, spm_addr, stride_bytes, 0U, 0U);
}

static void
launch_mpu_episode(uint32_t device_id,
                   uintptr_t a_addr, uintptr_t b_addr, uintptr_t c_addr,
                   uint32_t m, uint32_t n, uint32_t k,
                   uint32_t a_index, uint32_t b_index, uint32_t c_index)
{
    launch_mpu_mvin(device_id, MPU_BUFFER_A, a_index, m, n, k, a_addr, k);
    launch_mpu_mvin(device_id, MPU_BUFFER_B, b_index, m, n, k, b_addr, n);
    launch_mpu_load(device_id, MPU_BUFFER_A, a_index, m, n, k);
    launch_mpu_load(device_id, MPU_BUFFER_B, b_index, m, n, k);
    launch_mpu_compute(device_id, m, n, k);
    launch_mpu_drain(device_id, c_index, m, n, k);
    launch_mpu_mvout(device_id, c_index, m, n, k, c_addr,
                     n * sizeof(int32_t));
}

static inline uint32_t
move_layout_mode_cfg(uint32_t src_mem_space, uint32_t dst_mem_space,
                     uint32_t src_cut_dim, uint32_t dst_cut_dim)
{
    return (src_mem_space & 0x1U) | ((dst_mem_space & 0x1U) << 1) |
           ((src_cut_dim & 0x3U) << 2) | ((dst_cut_dim & 0x3U) << 4);
}

static inline void
launch_dma_move_layout(uintptr_t src_base, uintptr_t dst_base,
                       uint32_t shape_h, uint32_t shape_w, uint32_t shape_c,
                       uint32_t src_stride_h, uint32_t src_stride_w,
                       uint32_t src_stride_c, uint32_t dst_stride_h,
                       uint32_t dst_stride_w, uint32_t dst_stride_c)
{
    NpuCmd cmd;
    const uint32_t src_mem_space =
        (src_base >= SPM_BASE) ? DMA_MEM_SPACE_SPM : DMA_MEM_SPACE_DRAM;
    const uint32_t dst_mem_space =
        (dst_base >= SPM_BASE) ? DMA_MEM_SPACE_SPM : DMA_MEM_SPACE_DRAM;
    const uint32_t op_code = ((MPU_DATA_TYPE_INT8 & 0x7U) << 5) |
                             ((DMA_MODE_MOVE_LAYOUT & 0x7U) << 2);

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_DMA);
    cmd.setDeviceId(DMA_DEVICE_ID);
    cmd.setOpCode(op_code);
    cmd.setSyncIndicator(0U);
    cmd.setSetIndicatorSns(0U);
    cmd.setSetIndicatorSnd(0U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, (uint32_t)src_base);
    cmd.setWord(2U, (uint32_t)dst_base);
    cmd.setWord(3U, shape_h);
    cmd.setWord(4U, shape_w);
    cmd.setWord(5U, shape_c);
    cmd.setWord(6U, src_stride_h);
    cmd.setWord(7U, src_stride_w);
    cmd.setWord(8U, src_stride_c);
    cmd.setWord(9U, dst_stride_h);
    cmd.setWord(10U, dst_stride_w);
    cmd.setWord(11U, dst_stride_c);
    cmd.setWord(12U, 0U);
    cmd.setWord(13U, move_layout_mode_cfg(src_mem_space, dst_mem_space,
                                          DMA_CUT_DIM_W, DMA_CUT_DIM_W));
    cmd.setWord(14U, 0U);
    cmd.setWord(15U, 0U);
    cmd.launchCmd();
}

static int
run_basic_like(uint32_t device_id, uintptr_t a_addr, uintptr_t b_addr,
               uintptr_t c_addr, const int8_t *a_values,
               const int8_t *b_values, uint32_t m, uint32_t n, uint32_t k,
               uint32_t a_index, uint32_t b_index, uint32_t c_index,
               int32_t preset_c_value)
{
    int32_t expected[64];

    fill_matrix_i8(a_addr, m, k, k, a_values);
    fill_matrix_i8(b_addr, k, n, n, b_values);
    fill_matrix_i32(c_addr, m, n, n * sizeof(int32_t), preset_c_value);
    compute_expected(a_values, b_values, m, n, k, expected);

    launch_mpu_episode(device_id, a_addr, b_addr, c_addr,
                       m, n, k, a_index, b_index, c_index);

    if (!spin_until_i32_match(c_addr, m, n, n * sizeof(int32_t), expected)) {
        return 0;
    }
    return matrix_matches_i32(c_addr, m, n, n * sizeof(int32_t), expected);
}

static int
run_basic_tile_flow(uint32_t device_id)
{
    const int8_t a_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int8_t b_values[12] = {1, 0, 2, -1, 3, 1, 2, 1, 0, 1, -2, 4};
    return run_basic_like(device_id, A0_SPM, B0_SPM, C0_SPM,
                          a_values, b_values, 2, 3, 4, 0, 0, 0, 0);
}

static int
run_output_stationary_basic(uint32_t device_id)
{
    const int8_t a_values[6] = {1, -2, 3, 4, 0, -1};
    const int8_t b_values[6] = {2, -1, 1, 3, -2, 4};
    return run_basic_like(device_id, A0_SPM, B0_SPM, C0_SPM,
                          a_values, b_values, 2, 2, 3, 0, 0, 0, 99);
}

static int
run_ab_auto_release(uint32_t device_id)
{
    const int8_t a0[4] = {1, 2, 3, 4};
    const int8_t b0[4] = {5, 6, 7, 8};
    const int8_t a1[4] = {-1, 1, 2, -2};
    const int8_t b1[4] = {2, 0, -3, 1};

    if (!run_basic_like(device_id, A0_SPM, B0_SPM, C0_SPM,
                        a0, b0, 2, 2, 2, 0, 0, 0, 0)) {
        return 0;
    }
    return run_basic_like(device_id, A1_SPM, B1_SPM, C1_SPM,
                          a1, b1, 2, 2, 2, 0, 0, 1, 0);
}

static int
run_compute_latency_and_drain_handoff(uint32_t device_id)
{
    const int8_t a_values[15] = {
        1, 2, 3, 4, 5,
        -1, 0, 1, 2, 3,
        2, -2, 1, 0, 4,
    };
    const int8_t b_values[10] = {
        1, 0,
        -1, 2,
        3, 1,
        0, -2,
        2, 4,
    };
    return run_basic_like(device_id, A0_SPM, B0_SPM, C0_SPM,
                          a_values, b_values, 3, 2, 5, 0, 0, 0, 0);
}

static int
run_drain_handoff_allows_next_compute(uint32_t device_id)
{
    const int8_t a0_values[4] = {1, 2, 3, 4};
    const int8_t b0_values[4] = {5, 6, 7, 8};
    const int8_t a1_values[4] = {-1, 1, 2, -2};
    const int8_t b1_values[4] = {2, 0, -3, 1};
    int32_t expected0[4];
    int32_t expected1[4];

    fill_matrix_i8(A0_SPM, 2, 2, 2, a0_values);
    fill_matrix_i8(B0_SPM, 2, 2, 2, b0_values);
    fill_matrix_i8(A1_SPM, 2, 2, 2, a1_values);
    fill_matrix_i8(B1_SPM, 2, 2, 2, b1_values);
    fill_matrix_i32(C0_SPM, 2, 2, 2 * sizeof(int32_t), -11);
    fill_matrix_i32(C1_SPM, 2, 2, 2 * sizeof(int32_t), 33);
    compute_expected(a0_values, b0_values, 2, 2, 2, expected0);
    compute_expected(a1_values, b1_values, 2, 2, 2, expected1);

    launch_mpu_mvin(device_id, MPU_BUFFER_A, 0U, 2, 2, 2, A0_SPM, 2);
    launch_mpu_mvin(device_id, MPU_BUFFER_B, 0U, 2, 2, 2, B0_SPM, 2);
    launch_mpu_load(device_id, MPU_BUFFER_A, 0U, 2, 2, 2);
    launch_mpu_load(device_id, MPU_BUFFER_B, 0U, 2, 2, 2);
    launch_mpu_compute(device_id, 2, 2, 2);
    launch_mpu_drain(device_id, 0U, 2, 2, 2);

    launch_mpu_mvin(device_id, MPU_BUFFER_A, 1U, 2, 2, 2, A1_SPM, 2);
    launch_mpu_mvin(device_id, MPU_BUFFER_B, 1U, 2, 2, 2, B1_SPM, 2);
    launch_mpu_load(device_id, MPU_BUFFER_A, 1U, 2, 2, 2);
    launch_mpu_load(device_id, MPU_BUFFER_B, 1U, 2, 2, 2);
    launch_mpu_compute(device_id, 2, 2, 2);
    launch_mpu_drain(device_id, 1U, 2, 2, 2);

    launch_mpu_mvout(device_id, 0U, 2, 2, 2, C0_SPM,
                     2 * sizeof(int32_t));
    launch_mpu_mvout(device_id, 1U, 2, 2, 2, C1_SPM,
                     2 * sizeof(int32_t));

    if (!spin_until_i32_match(C0_SPM, 2, 2, 2 * sizeof(int32_t),
                              expected0)) {
        return 0;
    }
    if (!spin_until_i32_match(C1_SPM, 2, 2, 2 * sizeof(int32_t),
                              expected1)) {
        return 0;
    }
    return matrix_matches_i32(C0_SPM, 2, 2, 2 * sizeof(int32_t), expected0) &&
           matrix_matches_i32(C1_SPM, 2, 2, 2 * sizeof(int32_t), expected1);
}

static int
run_prefetch_compute_overlap(uint32_t device_id)
{
    const int8_t a0_values[4] = {1, 2, 3, 4};
    const int8_t b0_values[4] = {2, -1, 0, 3};
    const int8_t a1_values[4] = {-2, 1, 4, 0};
    const int8_t b1_values[4] = {1, 3, -1, 2};
    int32_t expected0[4];
    int32_t expected1[4];

    fill_matrix_i8(A0_SPM, 2, 2, 2, a0_values);
    fill_matrix_i8(B0_SPM, 2, 2, 2, b0_values);
    fill_matrix_i32(C0_SPM, 2, 2, 2 * sizeof(int32_t), -77);
    compute_expected(a0_values, b0_values, 2, 2, 2, expected0);

    fill_matrix_i8(A1_SPM, 2, 2, 2, a1_values);
    fill_matrix_i8(B1_SPM, 2, 2, 2, b1_values);
    fill_matrix_i32(C1_SPM, 2, 2, 2 * sizeof(int32_t), 55);
    compute_expected(a1_values, b1_values, 2, 2, 2, expected1);

    launch_mpu_episode(device_id, A0_SPM, B0_SPM, C0_SPM,
                       2, 2, 2, 0, 0, 0);
    launch_mpu_episode(device_id, A1_SPM, B1_SPM, C1_SPM,
                       2, 2, 2, 1, 1, 1);

    if (!spin_until_i32_match(C0_SPM, 2, 2, 2 * sizeof(int32_t), expected0)) {
        return 0;
    }
    if (!spin_until_i32_match(C1_SPM, 2, 2, 2 * sizeof(int32_t), expected1)) {
        return 0;
    }
    return matrix_matches_i32(C0_SPM, 2, 2, 2 * sizeof(int32_t), expected0) &&
           matrix_matches_i32(C1_SPM, 2, 2, 2 * sizeof(int32_t), expected1);
}

static int
run_spm_backpressure(uint32_t device_id)
{
    int8_t a_values[32];
    int8_t b_values[32];

    for (uint32_t i = 0; i < 32; ++i) {
        a_values[i] = (int8_t)((i % 7) - 3);
        b_values[i] = (int8_t)((i % 5) - 2);
    }
    return run_basic_like(device_id, A0_SPM, B0_SPM, C0_SPM,
                          a_values, b_values, 4, 4, 8, 0, 0, 0, 0);
}

static int
run_dma_spm_mpu_chain(uint32_t device_id)
{
    const int8_t a_values[6] = {1, 2, 3, 4, 5, 6};
    const int8_t b_values[6] = {-1, 0, 2, 3, 1, -2};
    int32_t expected[4];

    fill_matrix_i8(DRAM_A0, 2, 3, 3, a_values);
    fill_matrix_i8(DRAM_B0, 3, 2, 2, b_values);
    clear_bytes(A0_SPM, 6);
    clear_bytes(B0_SPM, 6);
    fill_matrix_i32(C0_SPM, 2, 2, 2 * sizeof(int32_t), 0);
    compute_expected(a_values, b_values, 2, 2, 3, expected);

    launch_dma_move_layout(DRAM_A0, A0_SPM, 2, 3, 1, 3, 1, 1, 3, 1, 1);
    launch_dma_move_layout(DRAM_B0, B0_SPM, 3, 2, 1, 2, 1, 1, 2, 1, 1);

    if (!spin_until_i8_match(A0_SPM, 2, 3, 3, a_values)) {
        return 0;
    }
    if (!spin_until_i8_match(B0_SPM, 3, 2, 2, b_values)) {
        return 0;
    }

    launch_mpu_episode(device_id, A0_SPM, B0_SPM, C0_SPM,
                       2, 2, 3, 0, 0, 0);
    if (!spin_until_i32_match(C0_SPM, 2, 2, 2 * sizeof(int32_t), expected)) {
        return 0;
    }
    return matrix_matches_i32(C0_SPM, 2, 2, 2 * sizeof(int32_t), expected);
}

static int
run_invalid_dtype(void)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_MPU);
    cmd.setDeviceId(0U);
    cmd.setOpCode(mpu_op_code(0x1U, MPU_OP_MVIN));
    cmd.clearCommonReservedBits();
    mpu_init_common_words(&cmd);
    cmd.setWord(MPU_WORD_BUFFER, mpu_buffer_word(MPU_BUFFER_A, 0U));
    cmd.setWord(MPU_WORD_M, 1U);
    cmd.setWord(MPU_WORD_N, 1U);
    cmd.setWord(MPU_WORD_K, 1U);
    cmd.setWord(MPU_WORD_SPM_ADDR_LO, (uint32_t)A0_SPM);
    cmd.setWord(MPU_WORD_STRIDE, 1U);
    cmd.launchCmd();
    return 1;
}

static int
run_invalid_opcode(void)
{
    mpu_launch_cmd(0U, mpu_op_code(MPU_DATA_TYPE_INT8, 0x1fU),
                   MPU_BUFFER_A, 0U, 1U, 1U, 1U, A0_SPM, 1U, 0U, 0U);
    return 1;
}

static int
run_invalid_mvin_c(void)
{
    mpu_launch_cmd(0U, mpu_op_code(MPU_DATA_TYPE_INT8, MPU_OP_MVIN),
                   MPU_BUFFER_C, 0U, 1U, 1U, 1U, C0_SPM, 4U, 0U, 0U);
    return 1;
}

static int
run_invalid_load_c(void)
{
    launch_mpu_load(0U, MPU_BUFFER_C, 0U, 1U, 1U, 1U);
    return 1;
}

static int
run_invalid_compute_without_loaded_inputs(void)
{
    launch_mpu_compute(0U, 1U, 1U, 1U);
    return 1;
}

static int
run_invalid_drain_without_ready_output(void)
{
    launch_mpu_drain(0U, 0U, 1U, 1U, 1U);
    return 1;
}

static int
run_invalid_mvout_without_full_c(void)
{
    launch_mpu_mvout(0U, 0U, 1U, 1U, 1U, C0_SPM, sizeof(int32_t));
    return 1;
}

static void
spin_for_panic_window(void)
{
    volatile uint32_t sink = 0;
    for (unsigned long iter = 0; iter < MAX_WAIT_ITERS; ++iter) {
        sink += *(volatile uint32_t *)SPM_BASE;
    }
    if (sink == 0xFFFFFFFFU) {
        printf("MPU_IMPOSSIBLE=%u\n", sink);
    }
}

int
main(int argc, char **argv)
{
    const char *scenario = argc > 1 ? argv[1] : "basic_tile_flow";
    int ok = 0;

    if (strcmp(scenario, "basic_tile_flow") == 0) {
        ok = run_basic_tile_flow(0U);
    } else if (strcmp(scenario, "output_stationary_basic") == 0) {
        ok = run_output_stationary_basic(0U);
    } else if (strcmp(scenario, "ab_auto_release_current_stage") == 0) {
        ok = run_ab_auto_release(0U);
    } else if (strcmp(scenario, "compute_latency_and_drain_handoff") == 0) {
        ok = run_compute_latency_and_drain_handoff(0U);
    } else if (strcmp(scenario, "drain_handoff_allows_next_compute") == 0) {
        ok = run_drain_handoff_allows_next_compute(0U);
    } else if (strcmp(scenario, "prefetch_compute_overlap") == 0) {
        ok = run_prefetch_compute_overlap(0U);
    } else if (strcmp(scenario, "spm_backpressure") == 0) {
        ok = run_spm_backpressure(0U);
    } else if (strcmp(scenario, "multi_instance_route") == 0) {
        ok = run_basic_tile_flow(1U);
    } else if (strcmp(scenario, "dma_spm_mpu_chain") == 0) {
        ok = run_dma_spm_mpu_chain(0U);
    } else if (strcmp(scenario, "invalid_dtype") == 0) {
        ok = run_invalid_dtype();
        spin_for_panic_window();
        printf("MPU_UNEXPECTED_NO_PANIC=%s\n", scenario);
        return 1;
    } else if (strcmp(scenario, "invalid_opcode") == 0) {
        ok = run_invalid_opcode();
        spin_for_panic_window();
        printf("MPU_UNEXPECTED_NO_PANIC=%s\n", scenario);
        return 1;
    } else if (strcmp(scenario, "mvin_c_rejected") == 0) {
        ok = run_invalid_mvin_c();
        spin_for_panic_window();
        printf("MPU_UNEXPECTED_NO_PANIC=%s\n", scenario);
        return 1;
    } else if (strcmp(scenario, "load_c_rejected") == 0) {
        ok = run_invalid_load_c();
        spin_for_panic_window();
        printf("MPU_UNEXPECTED_NO_PANIC=%s\n", scenario);
        return 1;
    } else if (strcmp(scenario, "compute_without_loaded_inputs") == 0) {
        ok = run_invalid_compute_without_loaded_inputs();
        spin_for_panic_window();
        printf("MPU_UNEXPECTED_NO_PANIC=%s\n", scenario);
        return 1;
    } else if (strcmp(scenario, "drain_without_ready_output") == 0) {
        ok = run_invalid_drain_without_ready_output();
        spin_for_panic_window();
        printf("MPU_UNEXPECTED_NO_PANIC=%s\n", scenario);
        return 1;
    } else if (strcmp(scenario, "mvout_without_full_c") == 0) {
        ok = run_invalid_mvout_without_full_c();
        spin_for_panic_window();
        printf("MPU_UNEXPECTED_NO_PANIC=%s\n", scenario);
        return 1;
    } else {
        printf("MPU_UNKNOWN_SCENARIO=%s\n", scenario);
        return 1;
    }

    if (!ok) {
        printf("MPU_SCENARIO_FAIL=%s\n", scenario);
        return 1;
    }

    printf("MPU_SCENARIO_PASS=%s\n", scenario);
    return 0;
}
