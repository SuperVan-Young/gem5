#include <stdint.h>
#include <stdlib.h>

#define CMD_BYTES 16UL
#define CMDQ_BASE 0x70000000UL
#define PORT_STRIDE (1UL << 20)
#define SYNC_BASE 0x71000000UL

#define DEVICE_TYPE_SYNC 0x1U
#define DEVICE_TYPE_SEU 0x2U
#define DEVICE_ID 0x0U

#define OP_SYNC_WAIT 0x0U
#define OP_SYNC_SET 0x1U
#define OP_SEU_EXEC 0x0U

static inline void
mmio_write32(uint64_t addr, uint32_t value)
{
    *(volatile uint32_t *)addr = value;
}

static inline uint32_t
cmd_header(uint32_t device_type, uint32_t device_id, uint32_t op_code,
           uint32_t payload_low16)
{
    return ((device_type & 0xFU) << 24) |
           ((device_id & 0xFU) << 20) |
           ((op_code & 0xFU) << 16) |
           (payload_low16 & 0xFFFFU);
}

static inline void
push_cmd(uint64_t port_base, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3)
{
    const uint64_t ctrl_addr = port_base + CMD_BYTES;
    mmio_write32(port_base + 0x0, w0);
    mmio_write32(port_base + 0x4, w1);
    mmio_write32(port_base + 0x8, w2);
    mmio_write32(port_base + 0xC, w3);
    mmio_write32(ctrl_addr, 0);
}

int
main(int argc, char **argv)
{
    uint32_t cpu_id = 0;
    uint32_t rounds = 64;

    if (argc > 1) {
        cpu_id = (uint32_t)strtoul(argv[1], NULL, 0);
    }
    if (argc > 2) {
        uint32_t parsed = (uint32_t)strtoul(argv[2], NULL, 0);
        if (parsed > 0) {
            rounds = parsed;
        }
    }

    const uint64_t port_base = CMDQ_BASE + ((uint64_t)cpu_id << 20);
    const uint32_t sync_idx = cpu_id & 0xFFU;

    for (uint32_t r = 0; r < rounds; ++r) {
        push_cmd(
            port_base,
            cmd_header(DEVICE_TYPE_SYNC, DEVICE_ID, OP_SYNC_WAIT, sync_idx),
            0x10000000U | (cpu_id << 8) | (r & 0xFFU),
            0x20000000U | (cpu_id << 8) | (r & 0xFFU),
            0x30000000U | (cpu_id << 8) | (r & 0xFFU));

        push_cmd(
            port_base,
            cmd_header(DEVICE_TYPE_SEU, DEVICE_ID, OP_SEU_EXEC,
                       ((cpu_id & 0xFFU) << 8) | (r & 0xFFU)),
            0x40000000U | (cpu_id << 16) | r,
            0x50000000U | (cpu_id << 16) | r,
            0x60000000U | (cpu_id << 16) | r);

        mmio_write32(SYNC_BASE,
                     cmd_header(DEVICE_TYPE_SYNC, DEVICE_ID,
                                OP_SYNC_SET, sync_idx));

        volatile uint32_t jitter = 0;
        for (uint32_t i = 0; i < ((cpu_id + 1U) * 16U); ++i) {
            jitter += (i ^ r);
        }
        (void)jitter;
    }

    for (;;) {
        asm volatile("" ::: "memory");
    }
}
