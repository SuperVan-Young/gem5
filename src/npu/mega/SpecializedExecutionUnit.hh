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
#include <optional>
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

/*
 * SpecializedExecutionUnit now models a generic SEU as:
 *
 *   MMIO staging -> dispatchQueue -> issueQueue -> uopQueue -> callback/event
 *
 * The base class owns all scheduling and transaction bookkeeping.
 * Subclasses are expected to provide command semantics only:
 *
 * 1. classify a macro command into load/exec/store,
 * 2. choose the target issue queue,
 * 3. emit the first batch of uops in prologue,
 * 4. use mem/event callbacks to append later uops or mark epilogue pending.
 *
 * Important execution rule:
 *
 * - uops do not need to be generated all at once in prologue.
 * - a callback may append more uops later.
 * - epilogue is not triggered just because the uop queue is temporarily empty.
 * - a callback must explicitly move the macro command to EpiloguePending.
 * - prologue/epilogue always assert the uop queue is empty when they run.
 */
class SpecializedExecutionUnit : public ClockedObject
{
  public:
    enum class Phase
    {
        Idle,
        Prologue,
        Active,
        EpiloguePending,
        Epilogue,
        Done,
    };

    enum class MacroCmdKind : uint8_t
    {
        Load = 0,
        Exec = 1,
        Store = 2,
    };

    enum class IssueQueueKind : uint8_t
    {
        Exec = 0,
        Mem = 1,
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

    /*
     * MicroOpContext describes an already materialized unit of work that is
     * ready to execute. The queue stores only uops that are ready right now.
     * Later callbacks may append more uops into the same queue.
     */
    struct MicroOpContext
    {
        enum class Kind
        {
            Load,
            Exec,
            Store,
            SyncWrite,
        };

        Kind kind = Kind::Exec;
        uint64_t macroCmdId = 0;
        uint32_t ownerIssueQueueId = 0;
        PortID portId = InvalidPortID;
        uint64_t token = 0;
        Addr addr = 0;
        size_t size = 0;
        Tick latency = 0;
        std::vector<uint8_t> data;
    };

    /*
     * MemTxnContext tracks in-flight packets so the base infrastructure can
     * route responses back to the owning macro command and issue queue.
     */
    struct MemTxnContext
    {
        enum class Kind
        {
            Load,
            Store,
            SyncWrite,
        };

        PacketPtr pkt = nullptr;
        Kind kind = Kind::Load;
        uint64_t macroCmdId = 0;
        uint32_t ownerIssueQueueId = 0;
        PortID portId = InvalidPortID;
        uint64_t token = 0;
        Addr addr = 0;
        size_t size = 0;
    };

    /*
     * MacroCmdContext is the lifecycle object for one macro command.
     * The base class owns dispatch, issue, callback routing, and completion.
     * Subclasses are expected to mutate this object only through command
     * semantic hooks and the protected helper methods below.
     */
    struct MacroCmdContext
    {
        uint64_t macroCmdId = 0;
        std::vector<uint8_t> cmd;
        CmdFields fields;
        MacroCmdKind kind = MacroCmdKind::Exec;
        uint32_t targetIssueQueueId = 0;
        Phase phase = Phase::Idle;
        bool waitingCallback = false;
        bool completionIssued = false;
        bool epilogueQueued = false;
        std::optional<PortID> boundMemPortId;
        std::deque<MicroOpContext> uopQueue;
    };

    /*
     * IssueQueueState models resource ownership. The queue and the mapped mem
     * port are intentionally kept separate: the queue represents scheduling
     * policy, while the optional mem port association represents one default
     * resource mapping strategy.
     */
    struct IssueQueueState
    {
        uint32_t issueQueueId = 0;
        IssueQueueKind kind = IssueQueueKind::Exec;
        std::deque<uint64_t> waitingMacroCmdIds;
        std::optional<uint64_t> activeMacroCmdId;
        std::optional<PortID> mappedMemPortId;
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
    uint64_t queueOccupancy() const;
    uint64_t completedCmdCount() const;
    bool isIssueBusy() const;
    uint64_t completedReadRespCount() const;
    uint64_t completedWriteRespCount() const;
    uint64_t completedIterationCount() const;
    uint64_t prologueCount() const;
    uint64_t executeCount() const;
    uint64_t epilogueCount() const;
    uint64_t maxActiveMicroOps() const;

  protected:
    virtual MacroCmdKind classifyMacroCmd(
        const std::vector<uint8_t> &cmd) const;
    virtual uint32_t classifyIssueQueue(const std::vector<uint8_t> &cmd,
                                        MacroCmdKind kind) const;
    virtual std::vector<IssueQueueState> buildIssueQueues() const;

    virtual void onMacroCmdBegin(MacroCmdContext &macroCmd);
    virtual void buildUops(MacroCmdContext &macroCmd);
    virtual void onMemUopComplete(MacroCmdContext &macroCmd,
                                  const MemTxnContext &txn, PacketPtr pkt);
    virtual void onExecUopComplete(MacroCmdContext &macroCmd,
                                   const MicroOpContext &uop);
    virtual void onMacroCmdEnd(MacroCmdContext &macroCmd);

    virtual bool buildCompletionSyncWord(const std::vector<uint8_t> &cmd,
                                         uint32_t &word) const;
    virtual void sendCompletionSyncWord(uint32_t word);

    void appendLoadUop(MacroCmdContext &macroCmd, Addr addr, size_t size);
    void appendStoreUop(MacroCmdContext &macroCmd, Addr addr, size_t size,
                        const std::vector<uint8_t> &data);
    void appendExecUop(MacroCmdContext &macroCmd, Tick latency);
    void markEpiloguePending(MacroCmdContext &macroCmd);

  protected:
    bool validMmioOffset(Addr offset, size_t size) const;
    bool writeDataBytes(Addr offset, const uint8_t *src, size_t size);
    bool writeDataChunk(Addr offset, PacketPtr pkt);
    bool canLaunchCmd() const;
    bool launchStagedCmd();
    bool handleRequest(PacketPtr pkt);
    void tryScheduleIssue();
    void issueOneCommand();
    bool handleMemResponse(PacketPtr pkt);

    uint32_t extractCmdWord(const std::vector<uint8_t> &cmd) const;
    uint32_t readCmdWord(const std::vector<uint8_t> &cmd,
                         size_t word_idx) const;
    CmdFields parseCmdFields(uint32_t word) const;
    AddrRangeList getAddrRanges() const;

    IssueQueueState &getIssueQueue(uint32_t issueQueueId);
    const IssueQueueState &getIssueQueue(uint32_t issueQueueId) const;
    bool issueQueueExists(uint32_t issueQueueId) const;
    PortID mappedMemPort(const MacroCmdContext &macroCmd) const;

    void dispatchCommands();
    void activateIssueQueues();
    void runPendingEpilogues();
    void issueReadyUops();
    void runMacroCmdPrologue(MacroCmdContext &macroCmd);
    void runMacroCmdEpilogue(MacroCmdContext &macroCmd);
    void finishMacroCmd(MacroCmdContext &macroCmd);
    void releaseIssueQueueOwner(uint32_t issueQueueId, uint64_t macroCmdId);
    void issueOneUop(MacroCmdContext &macroCmd, MicroOpContext uop);
    void issueMemUop(MacroCmdContext &macroCmd, MicroOpContext uop);
    void issueExecUop(MacroCmdContext &macroCmd, MicroOpContext uop);
    void finishExecution(uint32_t issueQueueId);
    void sendTrackedPacket(const MemTxnContext &txn,
                           const std::vector<uint8_t> *data = nullptr);
    void updateConcurrentMicroOps();
    bool hasSchedulableWork() const;

    CPUSidePort cpuSidePort;
    StagingBuffer stagingBuffer;

    const uint32_t macroCmdBytes;
    const uint32_t cmdQueueDepth;
    const Addr baseAddr;
    const bool syncEnqueueOnDataWrite;
    Tick debugProcessLatency;

    std::vector<std::unique_ptr<MemSidePort>> memSidePorts;
    std::deque<uint64_t> dispatchQueue;
    std::deque<uint64_t> pendingEpilogueCmdIds;
    std::vector<IssueQueueState> issueQueues;
    std::unordered_map<uint64_t, MacroCmdContext> macroCmdContexts;
    std::unordered_map<PacketPtr, MemTxnContext> activeMemTxns;
    std::unordered_map<uint32_t, MicroOpContext> activeExecUops;
    std::unordered_map<uint32_t, std::unique_ptr<EventFunctionWrapper>>
        execCompletionEvents;

    uint64_t nextMacroCmdId = 1;
    uint64_t nextMicroOpToken = 1;
    uint64_t completedCount = 0;
    uint64_t completedReadCount = 0;
    uint64_t completedWriteCount = 0;
    uint64_t completedIterationValue = 0;
    uint64_t prologueCountValue = 0;
    uint64_t executeCountValue = 0;
    uint64_t epilogueCountValue = 0;
    uint64_t maxConcurrentMicroOpsValue = 0;

    EventFunctionWrapper issueEvent;

    MemSidePort &getMemSidePort(PortID idx);
    const MemSidePort &getMemSidePort(PortID idx) const;

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
