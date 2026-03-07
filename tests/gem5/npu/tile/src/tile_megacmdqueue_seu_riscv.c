#include <stdint.h>

#define CMDQ_BASE 0x70000000UL
#define CMD_BYTES 16UL
#define CTRL_ADDR (CMDQ_BASE + CMD_BYTES)

#define DEVICE_TYPE 0x1U
#define DEVICE_ID 0x0U
#define NUM_CMDS 6

static inline void
mmio_write32(uint64_t addr, uint32_t value)
{
    *(volatile uint32_t *)addr = value;
}

static inline void
emit_command(uint32_t seq)
{
    uint32_t header = (DEVICE_TYPE << 24) | (DEVICE_ID << 20) | (seq & 0xFFFFF);
    mmio_write32(CMDQ_BASE + 0x0, header);
    mmio_write32(CMDQ_BASE + 0x4, 0xA0000000U | seq);
    mmio_write32(CMDQ_BASE + 0x8, 0xB0000000U | seq);
    mmio_write32(CMDQ_BASE + 0xC, 0xC0000000U | seq);
    mmio_write32(CTRL_ADDR, 0);
}

int
main(void)
{
    for (uint32_t i = 0; i < NUM_CMDS; ++i) {
        emit_command(i);
    }

    volatile uint64_t spin = 0;
    for (uint64_t i = 0; i < 2000000ULL; ++i) {
        spin += i;
    }

    return (int)(spin & 0);
}
