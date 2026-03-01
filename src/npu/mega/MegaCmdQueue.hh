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

#ifndef __NPU_MEGA_MEGA_CMD_QUEUE_HH__
#define __NPU_MEGA_MEGA_CMD_QUEUE_HH__

#include <cstdint>
#include <deque>
#include <vector>

#include "base/addr_range.hh"
#include "mem/port.hh"
#include "params/MegaCmdQueue.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class MegaCmdQueue : public SimObject
{
  private:
    class CPUSidePort : public ResponsePort
    {
      private:
        MegaCmdQueue *owner;
        int id;
        bool needRetry;
        PacketPtr blockedRespPacket;

      public:
        CPUSidePort(const std::string &name, int id, MegaCmdQueue *owner);

        void trySendRetry();

      protected:
        Tick recvAtomic(PacketPtr pkt) override
        {
            panic("MegaCmdQueue does not support recvAtomic");
        }

        bool recvTimingReq(PacketPtr pkt) override;

        void recvFunctional(PacketPtr pkt) override
        {
            panic("MegaCmdQueue does not support recvFunctional");
        }

        void recvRespRetry() override;

        AddrRangeList getAddrRanges() const override;
    };

    struct StagingBuffer
    {
        std::vector<uint8_t> bytes;
        size_t writeOffset = 0;
    };

    std::vector<CPUSidePort> cpuSidePorts;
    std::vector<StagingBuffer> stagingBuffers;
    std::deque<std::vector<uint8_t>> queue;

    const uint32_t numInputPort;
    const uint32_t megaCmdWidth;
    const uint32_t cmdQueueDepth;
    const uint32_t megaCmdBytes;

    static constexpr Addr DoorbellOffset = 0x1000;
    static constexpr Addr MaxAddrSpace = static_cast<Addr>(-1);

    bool canAcceptDoorbell() const;
    bool appendDataChunk(PortID port_id, PacketPtr pkt);
    bool enqueueStagedCommand(PortID port_id);
    bool handleRequest(PacketPtr pkt, PortID port_id);

    bool appendDataBytes(PortID port_id, const uint8_t *src, size_t size);

    AddrRangeList getAddrRanges() const;
    void trySendRetries();

  public:
    MegaCmdQueue(const MegaCmdQueueParams &params);

    void init() override;

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

    bool testWriteWord(uint64_t data_addr, uint32_t value);
    bool testRingDoorbell();
    bool popCmd();
    uint64_t queueOccupancy() const;
};

} // namespace gem5

#endif // __NPU_MEGA_MEGA_CMD_QUEUE_HH__
