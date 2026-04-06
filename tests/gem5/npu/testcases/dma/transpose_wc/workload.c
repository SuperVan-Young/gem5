#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Transpose a 3x2x4 tensor inside DRAM into a 3x4x2 tensor by swapping
     * the W and C dimensions, then verify the remapped coordinates.
     */
    return scenario_transpose_wc();
}
