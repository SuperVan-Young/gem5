/*
 * Copyright (c) 2026
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of their
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "cmd/common.hh"

enum SeuMultiOpcode
{
    SEU_MULTI_OP_EXEC = 0x0U,
};

enum SeuMultiLayout
{
    SEU_MULTI_NUM_PORTS = 4U,
    SEU_MULTI_SLOT_STRIDE_BYTES = 0x40U,
    SEU_MULTI_REPETITION = 3U,
    SEU_MULTI_CMD_DEVICE_ID = 0x0U,
    SEU_MULTI_CMD_SYNC = 0x42U,
    SEU_MULTI_CMD_READ_MASK = 0x0000000BU,  /* ports 0, 1, 3 */
    SEU_MULTI_CMD_WRITE_MASK = 0x0000000EU, /* ports 1, 2, 3 */
    SEU_MULTI_CMD_RESERVED = 0x00000000U,
};

static uint32_t
popcount32(uint32_t value)
{
    uint32_t count = 0U;

    while (value != 0U) {
        count += value & 1U;
        value >>= 1U;
    }

    return count;
}

static volatile uint32_t *
spm_slot_ptr(uint32_t port_id)
{
    return (volatile uint32_t *)(uintptr_t)(
        0x60000000UL + ((uint64_t)port_id * SEU_MULTI_SLOT_STRIDE_BYTES));
}

static uint32_t
initial_slot_value(uint32_t port_id)
{
    return 0x00010011U + (port_id * 0x00011111U);
}

static uint32_t
mix_slot_value(uint32_t current, uint32_t signature, uint32_t iteration,
               uint32_t port_id)
{
    return current + signature + ((iteration + 1U) * 0x10U) + (port_id + 1U);
}

static void
seed_spm_slots(uint32_t *expected_slots)
{
    for (uint32_t port = 0U; port < SEU_MULTI_NUM_PORTS; ++port) {
        const uint32_t value = initial_slot_value(port);

        expected_slots[port] = value;
        *spm_slot_ptr(port) = value;
    }
}

static uint32_t
build_iteration_signature(const uint32_t *slots, uint32_t read_mask)
{
    uint32_t signature = 0U;

    for (uint32_t port = 0U; port < SEU_MULTI_NUM_PORTS; ++port) {
        if ((read_mask & (1U << port)) != 0U) {
            signature += slots[port];
        }
    }

    return signature;
}

static void
advance_expected_slots(uint32_t *slots, uint32_t read_mask,
                       uint32_t write_mask, uint32_t iteration)
{
    const uint32_t signature = build_iteration_signature(slots, read_mask);

    for (uint32_t port = 0U; port < SEU_MULTI_NUM_PORTS; ++port) {
        if ((write_mask & (1U << port)) != 0U) {
            slots[port] = mix_slot_value(slots[port], signature, iteration,
                                         port);
        }
    }
}

static void
simulate_expected_result(uint32_t *slots, uint32_t read_mask,
                         uint32_t write_mask, uint32_t repetition)
{
    for (uint32_t iteration = 0U; iteration < repetition; ++iteration) {
        advance_expected_slots(slots, read_mask, write_mask, iteration);
    }
}

static int
wait_for_expected_slots(const uint32_t *expected_slots, uint64_t timeout)
{
    for (uint64_t spin = 0ULL; spin < timeout; ++spin) {
        int matched = 1;

        for (uint32_t port = 0U; port < SEU_MULTI_NUM_PORTS; ++port) {
            const uint32_t actual = *spm_slot_ptr(port);

            if (actual != expected_slots[port]) {
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
print_slot_snapshot(const char *prefix, const uint32_t *slots)
{
    printf("%s", prefix);
    for (uint32_t port = 0U; port < SEU_MULTI_NUM_PORTS; ++port) {
        printf(" P%u=%#x", port, slots[port]);
    }
    printf("\n");
}

static void
launch_seu_multiport_cmd(void)
{
    NpuCmd cmd;

    cmd.clear();
    cmd.setDeviceType(NPU_DEVICE_TYPE_VPU);
    cmd.setDeviceId(SEU_MULTI_CMD_DEVICE_ID);
    cmd.setOpCode(SEU_MULTI_OP_EXEC);
    cmd.setSyncIndicator(SEU_MULTI_CMD_SYNC);
    cmd.setSetIndicatorSns(1U);
    cmd.clearCommonReservedBits();
    cmd.setWord(1U, SEU_MULTI_CMD_READ_MASK);
    cmd.setWord(2U, SEU_MULTI_CMD_WRITE_MASK);
    cmd.setWord(3U, SEU_MULTI_REPETITION);
    cmd.setWord(4U, SEU_MULTI_CMD_RESERVED);
    cmd.launchCmd();
}

int
main(void)
{
    uint32_t expected_slots[SEU_MULTI_NUM_PORTS];
    const uint32_t read_mask = SEU_MULTI_CMD_READ_MASK;
    const uint32_t write_mask = SEU_MULTI_CMD_WRITE_MASK;
    const uint32_t repetition = SEU_MULTI_REPETITION;
    const uint32_t expected_read_replies =
        popcount32(read_mask) * repetition;
    const uint32_t expected_write_replies =
        popcount32(write_mask) * repetition;
    const uint32_t expected_prologues = repetition;
    const uint32_t expected_executes = repetition;
    const uint32_t expected_epilogues = repetition;
    const uint32_t expected_completed_cmds = 1U;

    seed_spm_slots(expected_slots);
    print_slot_snapshot("SEU_MULTI_INITIAL", expected_slots);

    launch_seu_multiport_cmd();

    simulate_expected_result(expected_slots, read_mask, write_mask,
                             repetition);

    if (wait_for_expected_slots(expected_slots, 40000000ULL) != 0) {
        uint32_t actual_slots[SEU_MULTI_NUM_PORTS];

        for (uint32_t port = 0U; port < SEU_MULTI_NUM_PORTS; ++port) {
            actual_slots[port] = *spm_slot_ptr(port);
        }

        print_slot_snapshot("SEU_MULTI_EXPECTED", expected_slots);
        print_slot_snapshot("SEU_MULTI_ACTUAL", actual_slots);
        printf("SEU_MULTI_TEST_FAIL\n");
        return 1;
    }

    print_slot_snapshot("SEU_MULTI_FINAL", expected_slots);
    printf("SEU_MULTI_EXPECTED_COMPLETED_CMDS=%u\n", expected_completed_cmds);
    printf("SEU_MULTI_EXPECTED_PROLOGUES=%u\n", expected_prologues);
    printf("SEU_MULTI_EXPECTED_EXECUTES=%u\n", expected_executes);
    printf("SEU_MULTI_EXPECTED_EPILOGUES=%u\n", expected_epilogues);
    printf("SEU_MULTI_EXPECTED_READ_REPLIES=%u\n", expected_read_replies);
    printf("SEU_MULTI_EXPECTED_WRITE_REPLIES=%u\n", expected_write_replies);
    printf("SEU_MULTI_EXPECTED_REPETITIONS=%u\n", repetition);
    return 0;
}
