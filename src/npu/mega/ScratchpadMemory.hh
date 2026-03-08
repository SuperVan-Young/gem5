/*
 * Copyright (c) 2026
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * ScratchpadMemory declaration
 *
 * A Scratchpad Memory (SPM) is a fast, software-managed SRAM-based
 * memory used for accelerator data caching. Unlike caches, SPM provides
 * deterministic access latency and explicit data management.
 */

#ifndef __NPU_MEGA_SCRATCHPAD_MEMORY_HH__
#define __NPU_MEGA_SCRATCHPAD_MEMORY_HH__

#include <list>

#include "mem/abstract_mem.hh"
#include "mem/port.hh"
#include "params/ScratchpadMemory.hh"

namespace gem5
{

namespace npu
{

/**
 * ScratchpadMemory implements a timing-mode SRAM-based scratchpad memory.
 *
 * Key features:
 * - Configurable access latency and bandwidth
 * - Timing mode simulation with proper backpressure handling
 * - Inherits from AbstractMemory for backing store management
 * - Single-port design for simplicity
 */
class ScratchpadMemory : public memory::AbstractMemory
{
  private:
    /**
     * Deferred packet stores a packet along with its scheduled
     * transmission time for timing mode response.
     */
    class DeferredPacket
    {
      public:
        const Tick tick;
        const PacketPtr pkt;

        DeferredPacket(PacketPtr _pkt, Tick _tick) : tick(_tick), pkt(_pkt)
        {}
    };

    /**
     * MemoryPort handles the response-side port interface for the SPM.
     * Receives requests from the CPU/accelerator and sends responses.
     */
    class MemoryPort : public ResponsePort
    {
      private:
        ScratchpadMemory& spm;

      public:
        MemoryPort(const std::string& _name, ScratchpadMemory& _spm);

      protected:
        Tick recvAtomic(PacketPtr pkt) override;
        Tick recvAtomicBackdoor(
                PacketPtr pkt, MemBackdoorPtr &_backdoor) override;
        void recvFunctional(PacketPtr pkt) override;
        void recvMemBackdoorReq(const MemBackdoorReq &req,
                MemBackdoorPtr &backdoor) override;
        bool recvTimingReq(PacketPtr pkt) override;
        void recvRespRetry() override;
        AddrRangeList getAddrRanges() const override;
    };

    MemoryPort port;

    /**
     * Fixed access latency for SRAM (in ticks).
     * SRAM typically has uniform read/write latency.
     */
    const Tick latency;

    /**
     * Bandwidth in ticks per byte.
     * Used to limit the acceptance rate of requests.
     */
    const double bandwidth;

    /**
     * Internal packet queue for timing mode.
     * Responses are queued here until ready to send.
     */
    std::list<DeferredPacket> packetQueue;

    /**
     * Track if the SPM is busy processing a request.
     * Used for bandwidth limiting.
     */
    bool isBusy;

    /**
     * Flag to retry an outstanding request that arrived while busy.
     */
    bool retryReq;

    /**
     * Flag indicating we failed to send a response and await retry.
     */
    bool retryResp;

    /**
     * Release the SPM after being busy and send retry if needed.
     */
    void release();

    /**
     * Event for releasing the SPM after bandwidth-limited access.
     */
    EventFunctionWrapper releaseEvent;

    /**
     * Dequeue a packet from the internal queue and send as response.
     */
    void dequeue();

    /**
     * Event for dequeuing and sending responses.
     */
    EventFunctionWrapper dequeueEvent;

    /**
     * Upstream caches need this packet until true is returned,
     * so hold it for deletion until a subsequent call.
     */
    std::unique_ptr<Packet> pendingDelete;

  public:
    ScratchpadMemory(const ScratchpadMemoryParams &p);

    DrainState drain() override;

    Port &getPort(const std::string &if_name,
                  PortID idx=InvalidPortID) override;
    void init() override;

  protected:
    Tick recvAtomic(PacketPtr pkt);
    Tick recvAtomicBackdoor(PacketPtr pkt, MemBackdoorPtr &_backdoor);
    void recvFunctional(PacketPtr pkt);
    void recvMemBackdoorReq(const MemBackdoorReq &req,
            MemBackdoorPtr &backdoor);
    bool recvTimingReq(PacketPtr pkt);
    void recvRespRetry();
};

} // namespace npu
} // namespace gem5

#endif // __NPU_MEGA_SCRATCHPAD_MEMORY_HH__
