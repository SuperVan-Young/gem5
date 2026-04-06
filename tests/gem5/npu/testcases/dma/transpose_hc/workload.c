#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Transpose a 2x3x4 tensor from SPM into a 4x3x2 tensor in DRAM by
     * swapping the H and C dimensions, then verify the remapped coordinates.
     */
    return scenario_transpose_hc();
}
