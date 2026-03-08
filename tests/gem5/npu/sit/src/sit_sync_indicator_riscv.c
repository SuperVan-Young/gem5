#include <stdint.h>

#define CMD_BYTES 16UL

#define CMDQ_BASE 0x70000000UL
#define CTRL_ADDR (CMDQ_BASE + CMD_BYTES)
#define SYNC_BASE 0x71000000UL

#define DEVICE_TYPE_SYNC 0x1U
#define DEVICE_TYPE_SEU 0x2U
#define DEVICE_ID 0x0U

#define OP_SYNC_WAIT 0x0U
#define OP_SYNC_SET 0x1U

#define SYNC_INDEX 7U

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
push_cmd(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3)
{
    mmio_write32(CMDQ_BASE + 0x0, w0);
    mmio_write32(CMDQ_BASE + 0x4, w1);
    mmio_write32(CMDQ_BASE + 0x8, w2);
    mmio_write32(CMDQ_BASE + 0xC, w3);
    mmio_write32(CTRL_ADDR, 0);
}

int
main(void)
{
    push_cmd(
        cmd_header(DEVICE_TYPE_SYNC, DEVICE_ID, OP_SYNC_WAIT, SYNC_INDEX),
        0x11111111U,
        0x22222222U,
        0x33333333U);

    push_cmd(
        cmd_header(DEVICE_TYPE_SEU, DEVICE_ID, 0x0U, 0x55AAU),
        0xAAA00001U,
        0xBBB00002U,
        0xCCC00003U);

    mmio_write32(SYNC_BASE, cmd_header(DEVICE_TYPE_SYNC, DEVICE_ID,
                                       OP_SYNC_SET, SYNC_INDEX));

    volatile uint64_t spin = 0;
    for (uint64_t i = 0; i < 4000000ULL; ++i) {
        spin += i;
    }

    return (int)(spin & 0);
}
