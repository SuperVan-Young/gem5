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
#include "mem/packet.hh"
#include "mem/port.hh"
#include "mem/request.hh"
#include "params/MegaCmdQueue.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class MegaCmdQueue : public ClockedObject
{
  private:
    class CPUSidePort : public ResponsePort
    {
      private:
        MegaCmdQueue *owner;
        PortID id;
        bool syncIndicatorPort;
        bool needRetry;
        PacketPtr blockedRespPacket;
        EventFunctionWrapper sendResponseEvent;

        void sendDeferredResponse();

      public:
        CPUSidePort(const std::string &name, PortID id, bool sync_indicator_port,
                    MegaCmdQueue *owner);

        void trySendRetry();
        void scheduleResponse();

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

    class MemSidePort : public RequestPort
    {
      private:
        MegaCmdQueue *owner;

      public:
        MemSidePort(const std::string &name, MegaCmdQueue *owner);

        bool sendPacket(PacketPtr pkt);

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;
        void recvRangeChange() override {}
    };

    struct StagingBuffer
    {
        std::vector<uint8_t> bytes;
    };

    struct CmdFields
    {
        uint8_t deviceType;
        uint8_t deviceId;
        uint8_t opCode;
        uint8_t syncIndicator;
        bool setIndicatorSns;
        bool setIndicatorSnd;
    };

    std::vector<CPUSidePort> cpuSidePorts;
    CPUSidePort syncIndicatorSidePort;
    MemSidePort memSidePort;
    std::vector<StagingBuffer> stagingBuffers;
    std::deque<std::vector<uint8_t>> queue;

    const uint32_t numInputPort;
    const uint32_t megaCmdWidth;
    const uint32_t cmdQueueDepth;
    const uint32_t megaCmdBytes;
    const Addr baseAddr;
    const Addr rangeAddr;
    const uint32_t numSyncIndicator;
    std::vector<uint8_t> syncIndicatorTable;
    std::vector<bool> pendingSyncDoneResponses;

    bool hasEnqueuedCmd;
    bool writeInFlight;
    bool writeAwaitingRetry;
    PacketPtr writePacket;
    EventFunctionWrapper clearEnqueueGateEvent;

    bool canPushMegaCmd() const;
    bool writeDataChunk(PortID port_id, Addr offset, PacketPtr pkt);
    bool recvTimingPushReq(PortID port_id);
    bool recvTimingPopReq();
    bool recvTimingSyncDoneReq(PortID port_id);
    bool handleRequest(PacketPtr pkt, PortID port_id);
    bool handleSyncIndicatorRequest(PacketPtr pkt);
    void clearEnqueueGate();

    bool writeDataBytes(PortID port_id, Addr offset, const uint8_t *src,
                        size_t size);
    bool validMmioOffset(Addr offset, size_t size) const;

    Addr portBaseAddr(PortID port_id) const;
    AddrRangeList getCpuAddrRanges(PortID port_id) const;
    AddrRangeList getSyncIndicatorAddrRanges() const;
    void trySendRetries();
    void tryCompleteSyncDoneResponses();
    void popMegaCmd();

    bool tryDispatchNext();
    void retryDispatch();
    bool handleMemResponse(PacketPtr pkt);
    Addr buildTargetAddr(const std::vector<uint8_t> &cmd) const;
    void cleanupWritePacket();
    uint32_t extractHeaderWord(const std::vector<uint8_t> &cmd) const;
    bool isDrainComplete() const;
    bool shouldDeferCpuResponse(PortID port_id) const;

    CmdFields parseCmdFields(const std::vector<uint8_t> &cmd) const;
    CmdFields parseCmdFields(uint32_t word) const;

  public:
    MegaCmdQueue(const MegaCmdQueueParams &params);
    ~MegaCmdQueue() override;

    void init() override;

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

    uint64_t queueOccupancy() const;
};

} // namespace gem5

#endif // __NPU_MEGA_MEGA_CMD_QUEUE_HH__
