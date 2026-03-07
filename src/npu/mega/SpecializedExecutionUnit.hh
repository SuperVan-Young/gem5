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

#ifndef __NPU_MEGA_SPECIALIZED_EXECUTION_UNIT_HH__
#define __NPU_MEGA_SPECIALIZED_EXECUTION_UNIT_HH__

#include <cstdint>
#include <deque>
#include <vector>

#include "base/addr_range.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "mem/request.hh"
#include "params/SpecializedExecutionUnit.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class SpecializedExecutionUnit : public ClockedObject
{
  private:
    class CPUSidePort : public ResponsePort
    {
      private:
        SpecializedExecutionUnit *owner;
        bool needRetry;
        PacketPtr blockedRespPacket;
        EventFunctionWrapper sendResponseEvent;

        void sendDeferredResponse();

      public:
        CPUSidePort(const std::string &name, SpecializedExecutionUnit *owner);

        void trySendRetry();
        void setNeedRetry() { needRetry = true; }

      protected:
        Tick recvAtomic(PacketPtr pkt) override
        {
            panic("SpecializedExecutionUnit does not support recvAtomic");
        }

        bool recvTimingReq(PacketPtr pkt) override;

        void recvFunctional(PacketPtr pkt) override
        {
            panic("SpecializedExecutionUnit does not support recvFunctional");
        }

        void recvRespRetry() override;

        AddrRangeList getAddrRanges() const override;
    };

    class MemSidePort : public RequestPort
    {
      private:
        SpecializedExecutionUnit *owner;
        PacketPtr blockedPacket;

      public:
        MemSidePort(const std::string &name, SpecializedExecutionUnit *owner);

        void sendPacket(PacketPtr pkt);

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;
        void recvRangeChange() override {}
    };

    struct StagingBuffer
    {
        std::vector<uint8_t> bytes;
    };

    CPUSidePort cpuSidePort;
    MemSidePort memSidePort;
    StagingBuffer stagingBuffer;
    std::deque<std::vector<uint8_t>> cmdQueue;

    const uint32_t macroCmdBytes;
    const uint32_t cmdQueueDepth;
    const Addr baseAddr;
    const bool syncEnqueueOnDataWrite;
    Tick debugProcessLatency;

    bool issueCmdBusy;
    uint64_t completedCount;
    PacketPtr activeMemPacket;
    uint8_t *activeMemBuffer;

    EventFunctionWrapper issueEvent;
    EventFunctionWrapper finishExecutionEvent;

    bool validMmioOffset(Addr offset, size_t size) const;
    bool writeDataBytes(Addr offset, const uint8_t *src, size_t size);
    bool writeDataChunk(Addr offset, PacketPtr pkt);
    bool canLaunchCmd() const;
    bool launchStagedCmd();
    bool handleRequest(PacketPtr pkt);
    void tryScheduleIssue();
    void issueOneCommand();
    void finishExecution();
    Tick process(const std::vector<uint8_t> &cmd);
    void cleanupActiveMemPacket();
    bool handleMemResponse(PacketPtr pkt);

    AddrRangeList getAddrRanges() const;

  public:
    SpecializedExecutionUnit(const SpecializedExecutionUnitParams &params);
    ~SpecializedExecutionUnit() override;

    void init() override;

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

    void setDebugProcessLatency(Tick latency);
    bool startBlockingRead(Addr addr, size_t size, uint8_t *buffer);
    bool startBlockingWrite(Addr addr, size_t size, const uint8_t *buffer);

    uint64_t queueOccupancy() const;
    uint64_t completedCmdCount() const;
    bool isIssueBusy() const;
};

} // namespace gem5

#endif // __NPU_MEGA_SPECIALIZED_EXECUTION_UNIT_HH__
