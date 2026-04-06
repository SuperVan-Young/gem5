#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Pre-fill a DRAM cache line with 0x11, issue one DMA fill command that
     * overwrites it with 0x5a, then verify the full 64B region.
     */
    return scenario_fill_nonzero_dram();
}
