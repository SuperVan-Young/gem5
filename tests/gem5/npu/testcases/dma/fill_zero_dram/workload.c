#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Pre-fill a DRAM cache line with non-zero bytes, issue one DMA fill
     * command that writes zeros, then verify the full 64B region.
     */
    return scenario_fill_zero_dram();
}
