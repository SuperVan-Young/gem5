#include "dma_functional_cases.hh"

int
main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /*
     * Queue two dependent DMA move commands back to back without an
     * explicit sync wait, then verify the final DRAM destination after
     * the queue drains.
     */
    return scenario_queued_chain();
}
