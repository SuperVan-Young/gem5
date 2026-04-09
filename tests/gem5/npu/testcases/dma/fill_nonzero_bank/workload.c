#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Fill DMA internal bank 1 with 64 bytes of 0x5a and wait on the
     * completion sync before exiting.
     */
    return scenario_fill_nonzero_bank();
}
