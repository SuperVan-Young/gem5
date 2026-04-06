#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Transpose a 2x4x3 tensor from DRAM into a 4x2x3 tensor in SPM by
     * swapping the H and W dimensions, then verify the remapped coordinates.
     */
    return scenario_transpose_hw();
}
