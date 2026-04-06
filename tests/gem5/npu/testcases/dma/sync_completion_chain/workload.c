#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Launch a DRAM->SPM DMA command that sets a completion sync, wait on that
     * sync, then launch a dependent SPM->DRAM DMA command and verify the end
     * result in DRAM.
     */
    return scenario_sync_completion();
}
