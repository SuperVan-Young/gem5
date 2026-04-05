/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "vpu.hh"

enum ElemwiseLayout
{
    VPU_DEVICE_ID = 0U,
    VPU_SLOT_STRIDE_BYTES = 0x40U,
    SRC0_PORT = 0U,
    SRC1_PORT = 1U,
    DST_PORT = 2U,
    ELEM_COUNT = 4U,
    INT_ADD_SYNC = 0x41U,
    INT_SUB_SYNC = 0x42U,
    INT_MUL_SYNC = 0x43U,
    INT_DIV_SYNC = 0x44U,
    FP_ADD_SYNC = 0x45U,
    FP_SUB_SYNC = 0x46U,
    FP_MUL_SYNC = 0x47U,
    FP_DIV_SYNC = 0x48U,
};

struct VpuStatsExpectation
{
    uint32_t completed_cmds;
    uint32_t prologues;
    uint32_t executes;
    uint32_t epilogues;
    uint32_t read_resps;
    uint32_t write_resps;
    uint32_t iterations;
};

static volatile uint32_t *
slot_word_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        0x60000000UL + ((uint64_t)port_id * VPU_SLOT_STRIDE_BYTES));
}

static uint32_t
float_to_bits(float value)
{
    uint32_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float
bits_to_float(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void
clear_slot(uint32_t port_id)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < VPU_SLOT_STRIDE_BYTES / sizeof(uint32_t);
         ++idx) {
        base[idx] = 0U;
    }
}

static void
store_u32_vector(uint32_t port_id, const uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        base[idx] = values[idx];
    }
}

static void
load_u32_vector(uint32_t port_id, uint32_t *values, uint32_t count)
{
    volatile uint32_t *base = slot_word_ptr(port_id);

    for (uint32_t idx = 0U; idx < count; ++idx) {
        values[idx] = base[idx];
    }
}

static int
wait_vector_match(uint32_t port_id, const uint32_t *expected,
                  uint32_t count, uint64_t timeout)
{
    uint32_t actual[ELEM_COUNT];

    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t idx = 0U; idx < count; ++idx) {
            actual[idx] = slot_word_ptr(port_id)[idx];
            if (actual[idx] != expected[idx]) {
                matched = 0;
                break;
            }
        }

        if (matched) {
            return 0;
        }
    }

    return -1;
}

static void
accumulate_expected_stats(struct VpuStatsExpectation *stats,
                          uint32_t repetition)
{
    stats->completed_cmds += 1U;
    stats->prologues += repetition;
    stats->executes += repetition;
    stats->epilogues += repetition;
    stats->read_resps += 2U * repetition;
    stats->write_resps += repetition;
    stats->iterations += repetition;
}

static int
run_int_case(uint32_t op_code, uint32_t sync_indicator,
             const uint32_t *lhs, const uint32_t *rhs, uint32_t *expected,
             uint32_t repetition, uint32_t dst_port)
{
    uint32_t actual[ELEM_COUNT];
    const uint32_t write_mask = 1U << dst_port;

    clear_slot(SRC0_PORT);
    clear_slot(SRC1_PORT);
    clear_slot(DST_PORT);
    store_u32_vector(SRC0_PORT, lhs, ELEM_COUNT);
    store_u32_vector(SRC1_PORT, rhs, ELEM_COUNT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        switch (op_code) {
          case VPU_OP_VADD:
            expected[idx] = lhs[idx] + rhs[idx];
            break;
          case VPU_OP_VSUB:
            expected[idx] = lhs[idx] - rhs[idx];
            break;
          case VPU_OP_VMUL:
            expected[idx] = lhs[idx] * rhs[idx];
            break;
          case VPU_OP_VDIV: {
            const int32_t lhs_i32 = (int32_t)lhs[idx];
            const int32_t rhs_i32 = (int32_t)rhs[idx];
            int32_t value = 0;
            if (rhs_i32 == 0) {
                value = 0;
            } else if (lhs_i32 == INT32_MIN && rhs_i32 == -1) {
                value = INT32_MAX;
            } else {
                value = lhs_i32 / rhs_i32;
            }
            memcpy(&expected[idx], &value, sizeof(uint32_t));
            break;
          }
          default:
            printf("VPU_ELEMWISE_FAIL bad_int_opcode=%u\n", op_code);
            return -1;
        }
    }

    vpu_cmd_launch_binary(VPU_DEVICE_ID, op_code, sync_indicator, 0x3U,
                          write_mask,
                          repetition, ELEM_COUNT, sizeof(uint32_t),
                          sizeof(uint32_t), VPU_DATA_I32);

    if (wait_vector_match(dst_port, expected, ELEM_COUNT, 60000000ULL) != 0) {
        load_u32_vector(dst_port, actual, ELEM_COUNT);
        printf("VPU_ELEMWISE_FAIL int op=%u", op_code);
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%u act[%u]=%u", idx, expected[idx], idx,
                   actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

static int
run_float_case(uint32_t op_code, uint32_t sync_indicator,
               const uint32_t *lhs_bits, const uint32_t *rhs_bits,
               uint32_t *expected_bits, uint32_t repetition,
               uint32_t dst_port)
{
    uint32_t actual[ELEM_COUNT];
    const uint32_t write_mask = 1U << dst_port;

    clear_slot(SRC0_PORT);
    clear_slot(SRC1_PORT);
    clear_slot(DST_PORT);
    store_u32_vector(SRC0_PORT, lhs_bits, ELEM_COUNT);
    store_u32_vector(SRC1_PORT, rhs_bits, ELEM_COUNT);
    for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
        const float lhs = bits_to_float(lhs_bits[idx]);
        const float rhs = bits_to_float(rhs_bits[idx]);
        float value = 0.0f;

        switch (op_code) {
          case VPU_OP_VADD:
            value = lhs + rhs;
            break;
          case VPU_OP_VSUB:
            value = lhs - rhs;
            break;
          case VPU_OP_VMUL:
            value = lhs * rhs;
            break;
          case VPU_OP_VDIV:
            value = lhs / rhs;
            break;
          default:
            printf("VPU_ELEMWISE_FAIL bad_fp_opcode=%u\n", op_code);
            return -1;
        }

        expected_bits[idx] = float_to_bits(value);
    }

    vpu_cmd_launch_binary(VPU_DEVICE_ID, op_code, sync_indicator, 0x3U,
                          write_mask,
                          repetition, ELEM_COUNT, sizeof(uint32_t),
                          sizeof(uint32_t), VPU_DATA_F32);

    if (wait_vector_match(dst_port, expected_bits, ELEM_COUNT,
                          60000000ULL) != 0) {
        load_u32_vector(dst_port, actual, ELEM_COUNT);
        printf("VPU_ELEMWISE_FAIL float op=%u", op_code);
        for (uint32_t idx = 0U; idx < ELEM_COUNT; ++idx) {
            printf(" exp[%u]=%#x act[%u]=%#x", idx, expected_bits[idx], idx,
                   actual[idx]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

int
main(void)
{
    const uint32_t int_lhs[ELEM_COUNT] = {10U, 20U, 30U, 40U};
    const uint32_t int_rhs[ELEM_COUNT] = {1U, 2U, 3U, 4U};
    const uint32_t int_div_lhs[ELEM_COUNT] = {
        (uint32_t)INT32_MIN, 21U, 9U, 0U
    };
    const uint32_t int_div_rhs[ELEM_COUNT] = {
        (uint32_t)-1, 0U, (uint32_t)-4, 5U
    };
    const uint32_t fp_lhs[ELEM_COUNT] = {
        float_to_bits(1.5f),
        float_to_bits(-2.0f),
        float_to_bits(4.0f),
        float_to_bits(0.5f),
    };
    const uint32_t fp_rhs[ELEM_COUNT] = {
        float_to_bits(2.0f),
        float_to_bits(0.5f),
        float_to_bits(-1.0f),
        float_to_bits(8.0f),
    };
    const uint32_t fp_div_lhs[ELEM_COUNT] = {
        float_to_bits(9.0f),
        float_to_bits(-3.0f),
        float_to_bits(1.0f),
        float_to_bits(0.0f),
    };
    const uint32_t fp_div_rhs[ELEM_COUNT] = {
        float_to_bits(3.0f),
        float_to_bits(0.0f),
        float_to_bits(0.0f),
        float_to_bits(4.0f),
    };
    uint32_t expected[ELEM_COUNT];
    struct VpuStatsExpectation stats = {0};

    if (run_int_case(VPU_OP_VADD, INT_ADD_SYNC, int_lhs, int_rhs, expected,
                     1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_int_case(VPU_OP_VSUB, INT_SUB_SYNC, int_lhs, int_rhs, expected,
                     2U, SRC0_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    if (run_int_case(VPU_OP_VMUL, INT_MUL_SYNC, int_lhs, int_rhs, expected,
                     1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_int_case(VPU_OP_VDIV, INT_DIV_SYNC, int_div_lhs, int_div_rhs,
                     expected, 1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_float_case(VPU_OP_VADD, FP_ADD_SYNC, fp_lhs, fp_rhs, expected,
                       1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_float_case(VPU_OP_VSUB, FP_SUB_SYNC, fp_lhs, fp_rhs, expected,
                       1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    if (run_float_case(VPU_OP_VMUL, FP_MUL_SYNC, fp_lhs, fp_rhs, expected,
                       2U, SRC0_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 2U);

    if (run_float_case(VPU_OP_VDIV, FP_DIV_SYNC, fp_div_lhs, fp_div_rhs,
                       expected, 1U, DST_PORT) != 0) {
        return 1;
    }
    accumulate_expected_stats(&stats, 1U);

    printf("VPU_ELEMWISE_EXPECTED_COMPLETED_CMDS=%u\n",
           stats.completed_cmds);
    printf("VPU_ELEMWISE_EXPECTED_PROLOGUES=%u\n", stats.prologues);
    printf("VPU_ELEMWISE_EXPECTED_EXECUTES=%u\n", stats.executes);
    printf("VPU_ELEMWISE_EXPECTED_EPILOGUES=%u\n", stats.epilogues);
    printf("VPU_ELEMWISE_EXPECTED_READ_RESPS=%u\n", stats.read_resps);
    printf("VPU_ELEMWISE_EXPECTED_WRITE_RESPS=%u\n", stats.write_resps);
    printf("VPU_ELEMWISE_EXPECTED_ITERATIONS=%u\n", stats.iterations);
    printf("VPU_ELEMWISE_PASS\n");
    return 0;
}
