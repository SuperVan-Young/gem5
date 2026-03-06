#include <stdint.h>

#define CMDQ_BASE 0x70000000UL
#define CMD_BYTES 16UL
#define CTRL_ADDR (CMDQ_BASE + CMD_BYTES)

static inline void
mmio_write32(uint64_t addr, uint32_t value)
{
    *(volatile uint32_t *)addr = value;
}

int
main(void)
{
    mmio_write32(CMDQ_BASE + 0x0, 0x1000);
    mmio_write32(CMDQ_BASE + 0x4, 0x1001);
    mmio_write32(CMDQ_BASE + 0x8, 0x1002);
    mmio_write32(CMDQ_BASE + 0xC, 0x1003);
    mmio_write32(CTRL_ADDR, 0);

    mmio_write32(CMDQ_BASE + 0x0, 0x2000);
    mmio_write32(CMDQ_BASE + 0x4, 0x2001);
    mmio_write32(CMDQ_BASE + 0x8, 0x2002);
    mmio_write32(CMDQ_BASE + 0xC, 0x2003);
    mmio_write32(CTRL_ADDR, 0);

    mmio_write32(CTRL_ADDR, 1);

    mmio_write32(CMDQ_BASE + 0x0, 0x3000);
    mmio_write32(CMDQ_BASE + 0x4, 0x3001);
    mmio_write32(CMDQ_BASE + 0x8, 0x3002);
    mmio_write32(CMDQ_BASE + 0xC, 0x3003);
    mmio_write32(CTRL_ADDR, 0);

    return 0;
}
