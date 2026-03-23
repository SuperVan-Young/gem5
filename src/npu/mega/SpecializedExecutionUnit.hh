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
#include <memory>
#include <unordered_map>
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
  public:
    enum class Phase
    {
        Idle,
        Prologue,
        LaunchingMvin,
        WaitingMvin,
        Executing,
        LaunchingMvout,
        WaitingMvout,
        Epilogue,
        Completing,
    };

    struct MemTxnContext
    {
        enum class Kind
        {
            Mvin,
            Mvout,
            SyncWrite,
        };

        PacketPtr pkt = nullptr;
        PortID portId = InvalidPortID;
        Kind kind = Kind::Mvin;
        uint64_t iteration = 0;
        uint64_t token = 0;
        Addr addr = 0;
        size_t size = 0;
    };

    struct MemRequestDesc
    {
        // Request description used by the base class to materialize packets.
        // Subclasses should treat this as a declarative description, not as a
        // live packet or transaction owner.
        PortID portId = InvalidPortID;
        MemTxnContext::Kind kind = MemTxnContext::Kind::Mvin;
        Addr addr = 0;
        size_t size = 0;
        std::vector<uint8_t> data;
        uint64_t iteration = 0;
        uint64_t token = 0;
    };

    struct CmdFields
    {
        uint8_t deviceType = 0;
        uint8_t deviceId = 0;
        uint8_t opCode = 0;
        uint8_t syncIndicator = 0;
        bool setIndicatorSns = false;
        bool setIndicatorSnd = false;
    };

    struct ActiveExecution
    {
        Phase phase = Phase::Idle;
        std::vector<uint8_t> cmd;
        CmdFields fields;
        uint32_t readMask = 0;
        uint32_t writeMask = 0;
        uint32_t repetition = 0;
        uint32_t reserved = 0;
        uint64_t iteration = 0;
        uint64_t completedIterations = 0;
        uint64_t prologueCount = 0;
        uint64_t executeCount = 0;
        uint64_t epilogueCount = 0;
        uint64_t completedReadRespCount = 0;
        uint64_t completedWriteRespCount = 0;
        std::vector<PortID> readPorts;
        std::vector<PortID> writePorts;
        std::unordered_map<PortID, std::vector<uint8_t>> readResults;
        std::unordered_map<PortID, std::vector<uint8_t>> writeResults;
    };

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
        PortID portId = InvalidPortID;

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;
        void recvRangeChange() override {}
    };

    struct StagingBuffer
    {
        std::vector<uint8_t> bytes;
    };

  public:
    // Port-aware buffer helpers for subclasses. The base class materializes
    // packets, selects the port, and tracks the transaction lifecycle.
    bool startBlockingRead(Addr addr, size_t size, uint8_t *buffer,
                           PortID port_id);
    bool startBlockingWrite(Addr addr, size_t size, const uint8_t *buffer,
                            PortID port_id);

    uint64_t queueOccupancy() const;
    uint64_t completedCmdCount() const;
    bool isIssueBusy() const;
    uint64_t completedReadRespCount() const;
    uint64_t completedWriteRespCount() const;
    uint64_t completedIterationCount() const;
    uint64_t prologueCount() const;
    uint64_t executeCount() const;
    uint64_t epilogueCount() const;

    virtual void onCommandBegin(ActiveExecution &exec) { (void)exec; }
    virtual void prologue(ActiveExecution &exec) { (void)exec; }
    virtual void buildMvinRequests(ActiveExecution &exec,
                                   std::vector<MemRequestDesc> &reqs)
    {
        (void)exec;
        (void)reqs;
    }
    virtual void onMvinResponse(ActiveExecution &exec,
                                const MemTxnContext &txn,
                                PacketPtr pkt)
    {
        (void)exec;
        (void)txn;
        (void)pkt;
    }
    virtual Tick execute(ActiveExecution &exec)
    {
        (void)exec;
        return debugProcessLatency;
    }
    virtual void buildMvoutRequests(ActiveExecution &exec,
                                    std::vector<MemRequestDesc> &reqs)
    {
        (void)exec;
        (void)reqs;
    }
    virtual void onMvoutResponse(ActiveExecution &exec,
                                 const MemTxnContext &txn,
                                 PacketPtr pkt)
    {
        (void)exec;
        (void)txn;
        (void)pkt;
    }
    virtual void epilogue(ActiveExecution &exec) { (void)exec; }
    virtual bool shouldExit(const ActiveExecution &exec) const
    {
        return exec.repetition == 0 ||
               exec.completedIterations >= exec.repetition;
    }

  protected:
    // Compatibility entry point; completion sync defaults to port 0.
    virtual void sendCompletionSyncWord(uint32_t word);
    virtual bool handleMemResponse(PacketPtr pkt);
    virtual bool buildCompletionSyncWord(const std::vector<uint8_t> &cmd,
                                         uint32_t &word) const;

  protected:
    bool validMmioOffset(Addr offset, size_t size) const;
    bool writeDataBytes(Addr offset, const uint8_t *src, size_t size);
    bool writeDataChunk(Addr offset, PacketPtr pkt);
    bool canLaunchCmd() const;
    bool launchStagedCmd();
    bool handleRequest(PacketPtr pkt);
    void tryScheduleIssue();
    void issueOneCommand();
    uint32_t extractCmdWord(const std::vector<uint8_t> &cmd) const;
    CmdFields parseCmdFields(uint32_t word) const;

    AddrRangeList getAddrRanges() const;

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
    std::vector<std::unique_ptr<MemSidePort>> memSidePorts;
    PacketPtr activeMemPacket;
    std::vector<uint8_t> activeCmd;
    ActiveExecution activeExecution;
    std::unordered_map<PacketPtr, MemTxnContext> activeMemTxns;
    uint64_t nextMemTxnToken = 0;

    EventFunctionWrapper issueEvent;
    EventFunctionWrapper finishExecutionEvent;

    // Legacy path retained for compatibility; prefer the port-aware buffer
    // helpers above.
    virtual void startExecuteCommand(const std::vector<uint8_t> &cmd);
    void sendCompletionSyncWord(uint32_t word, PortID port_id);

    void finishExecution();
    Tick process(const std::vector<uint8_t> &cmd);
    void completeActiveCommand();
    void sendMemRequest(PacketPtr pkt);
    void sendMemRequest(PacketPtr pkt, PortID port_id);
    bool startBlockingRead(Addr addr, size_t size, uint8_t *buffer);
    bool startBlockingWrite(Addr addr, size_t size, const uint8_t *buffer);
    void cleanupActiveMemPacket();
    MemSidePort &getMemSidePort(PortID idx);
    const MemSidePort &getMemSidePort(PortID idx) const;
    void beginActiveCommand(ActiveExecution &exec);
    void advanceActivePhase(ActiveExecution &exec);
    void runProloguePhase(ActiveExecution &exec);
    void launchMvinPhase(ActiveExecution &exec);
    bool handleMvinResponseInternal(ActiveExecution &exec, PacketPtr pkt);
    void scheduleExecutePhase(ActiveExecution &exec);
    void finishExecutePhase(ActiveExecution &exec);
    void launchMvoutPhase(ActiveExecution &exec);
    bool handleMvoutResponseInternal(ActiveExecution &exec, PacketPtr pkt);
    void runEpiloguePhase(ActiveExecution &exec);
    void finalizeActiveCommand(ActiveExecution &exec);

  public:
    SpecializedExecutionUnit(const SpecializedExecutionUnitParams &params);
    ~SpecializedExecutionUnit() override;

    void init() override;

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

    void setDebugProcessLatency(Tick latency);
};

} // namespace gem5

#endif // __NPU_MEGA_SPECIALIZED_EXECUTION_UNIT_HH__
