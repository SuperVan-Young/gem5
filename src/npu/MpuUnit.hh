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

#ifndef __NPU_MPU_UNIT_HH__
#define __NPU_MPU_UNIT_HH__

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/statistics.hh"
#include "npu/SpecializedExecutionUnit.hh"
#include "params/MpuUnit.hh"

namespace gem5
{

class MpuUnit : public SpecializedExecutionUnit
{
  public:
    enum class CmdKind : uint8_t
    {
        Mvin = 0x0,
        Mvout = 0x1,
        Load = 0x2,
        Compute = 0x3,
        Drain = 0x4,
        ComputeFused = 0x5,
    };

    enum class BufferKind : uint8_t
    {
        A = 0x0,
        B = 0x1,
        C = 0x2,
        Reserved = 0x3,
    };

    enum class BufferState : uint8_t
    {
        Empty = 0,
        LoadingFromSpm = 1,
        Full = 2,
        LoadedToInput = 3,
        DrainingToBuffer = 4,
        WritingToSpm = 5,
    };

    enum class OutputStorageState : uint8_t
    {
        Empty = 0,
        Active = 1,
        ReadyToDrain = 2,
    };

    struct ParsedCmd
    {
        CmdFields header;
        uint8_t dataType = 0;
        CmdKind kind = CmdKind::Mvin;
        BufferKind bufferKind = BufferKind::Reserved;
        uint8_t bufferIndex = 0;
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        Addr spmAddr = 0;
        uint32_t strideBytes = 0;
        uint32_t flags = 0;
    };

  private:
    struct ABBufferSlot
    {
        BufferState state = BufferState::Empty;
        uint32_t rows = 0;
        uint32_t cols = 0;
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        std::vector<int8_t> data;

        void reset();
    };

    struct CBufferSlot
    {
        BufferState state = BufferState::Empty;
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        std::vector<int32_t> data;

        void reset();
    };

    struct LoadedInputContext
    {
        bool valid = false;
        uint8_t bufferIndex = 0;
        uint32_t rows = 0;
        uint32_t cols = 0;
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;

        void reset();
    };

    struct OutputStorage
    {
        OutputStorageState state = OutputStorageState::Empty;
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        Tick readyTick = 0;
        std::vector<int32_t> data;

        void reset();
    };

    struct QueueEntry
    {
        CmdKind kind = CmdKind::Mvin;
        BufferKind bufferKind = BufferKind::Reserved;
        uint8_t bufferIndex = 0;
        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
    };

    struct ScoreboardState
    {
        bool loadedAReady = false;
        bool loadedBReady = false;
        bool outputStorageActive = false;
        bool outputStorageReady = false;
        std::array<bool, 2> cDrainReserved = {false, false};
        std::array<bool, 2> cWritebackBusy = {false, false};
    };

    struct PendingMemWindow
    {
        bool active = false;
        Tick startTick = 0;
        Tick lastRespTick = 0;
    };

    enum class PendingExecAction : uint8_t
    {
        None = 0,
        LoadRetry,
        LoadCommit,
    };

    struct MpuMacroRuntime
    {
        ParsedCmd parsed;
        Tick commandStartTick = 0;
        PendingMemWindow memWindow;
        uint32_t nextMemRow = 0;
        PendingExecAction pendingExecAction = PendingExecAction::None;
    };

    struct MpuStats : public statistics::Group
    {
        MpuStats(statistics::Group *parent);

        statistics::Scalar mvinCmdCount;
        statistics::Scalar loadCmdCount;
        statistics::Scalar computeCmdCount;
        statistics::Scalar drainCmdCount;
        statistics::Scalar mvoutCmdCount;

        statistics::Scalar totalABytesIn;
        statistics::Scalar totalBBytesIn;
        statistics::Scalar totalCBytesOut;
        statistics::Scalar totalOutputElementsDrained;
        statistics::Scalar totalMacOps;

        statistics::Scalar busyCycles;
        statistics::Scalar idleCycles;
        statistics::Scalar stallCyclesWaitingForSpm;
        statistics::Scalar stallCyclesBufferHazard;
        statistics::Scalar stallCyclesOutputStorageUnavailable;
        statistics::Scalar stallCyclesDrainDestBusy;
    } stats;

    static constexpr uint8_t MpuDeviceType = 0x3;
    static constexpr uint8_t Int8DataType = 0x0;
    static constexpr size_t CacheLineBytes = 64;
    static constexpr Addr SpmBase = 0x60000000ULL;
    static constexpr Addr SpmEnd = 0x6fffffffULL;

    const uint32_t arrayDim;
    const uint32_t aBufferCapacityBytes;
    const uint32_t bBufferCapacityBytes;
    const uint32_t cBufferCapacityBytes;
    const uint32_t memUopQueueDepth;
    const uint32_t execUopQueueDepth;
    const uint32_t drainUopQueueDepth;
    const Tick mvinRequestLatency;
    const Tick mvoutRequestLatency;
    const Tick loadLatencyBase;
    const Tick drainLatencyBase;
    const uint32_t numMpuMemSidePorts;

    std::array<ABBufferSlot, 2> aBuffers;
    std::array<ABBufferSlot, 2> bBuffers;
    std::array<CBufferSlot, 2> cBuffers;
    LoadedInputContext loadedA;
    LoadedInputContext loadedB;
    OutputStorage outputStorage;
    ScoreboardState scoreboard;

    std::unordered_map<uint64_t, QueueEntry> macroCmdFifo;
    std::unordered_map<uint64_t, QueueEntry> memUopQueue;
    std::unordered_map<uint64_t, QueueEntry> execUopQueue;
    std::unordered_map<uint64_t, QueueEntry> drainUopQueue;

    Tick busyStateChangeTick = 0;
    bool busyStateKnown = false;
    bool busyState = false;

    uint64_t lastCommandLatencyCyclesValue = 0;
    uint64_t lastComputeLatencyCyclesValue = 0;
    uint64_t lastDrainLatencyCyclesValue = 0;
    uint64_t lastOutputReadyLatencyCyclesValue = 0;

    std::unordered_map<uint64_t, MpuMacroRuntime> macroRuntimes;

    uint32_t extractWord(const std::vector<uint8_t> &cmd, size_t index) const;
    ParsedCmd parseCommand(const std::vector<uint8_t> &cmd) const;
    void validateCommand(const std::vector<uint8_t> &rawCmd,
                         const ParsedCmd &cmd) const;
    void validateMvin(const ParsedCmd &cmd) const;
    void validateLoadStatic(const ParsedCmd &cmd) const;
    void validateCompute(const ParsedCmd &cmd) const;
    void validateComputeFused(const ParsedCmd &cmd) const;
    void validateDrain(const ParsedCmd &cmd) const;
    void validateMvout(const ParsedCmd &cmd) const;
    bool loadReady(const ParsedCmd &cmd) const;
    bool fusedComputeReady(const ParsedCmd &cmd) const;

    void resetCommandStructures();
    void pushQueueEntry(uint64_t macroCmdId, const ParsedCmd &cmd);
    void popQueueEntry(uint64_t macroCmdId, const ParsedCmd &cmd);
    void refreshScoreboard();
    void updateBusyAccounting(bool now_busy);
    uint64_t elapsedCyclesSince(Tick start) const;
    uint64_t currentBusyCycles() const;
    uint64_t currentIdleCycles() const;

    ABBufferSlot &selectedABuffer(BufferKind kind, uint8_t index);
    const ABBufferSlot &selectedABuffer(BufferKind kind, uint8_t index) const;
    CBufferSlot &selectedCBuffer(uint8_t index);
    const CBufferSlot &selectedCBuffer(uint8_t index) const;

    uint32_t expectedRows(const ParsedCmd &cmd) const;
    uint32_t expectedCols(const ParsedCmd &cmd) const;
    uint32_t expectedRowBytes(const ParsedCmd &cmd) const;
    uint32_t requiredBytes(const ParsedCmd &cmd) const;
    bool isContiguousSpmWindow(const ParsedCmd &cmd) const;
    PortID mvinPortId(BufferKind kind) const;
    PortID mvinAPortId() const;
    PortID mvinBPortId() const;
    PortID mvoutPortId() const;
    void validateSpmWindow(const ParsedCmd &cmd) const;
    void beginMemWindow(MpuMacroRuntime &runtime);
    void observeMemResponse(MpuMacroRuntime &runtime);
    void finalizeMemWindow(MpuMacroRuntime &runtime);
    MpuMacroRuntime &runtimeFor(uint64_t macroCmdId);
    const MpuMacroRuntime &runtimeFor(uint64_t macroCmdId) const;
    void appendMvinRowUop(MacroCmdContext &macroCmd, MpuMacroRuntime &runtime);
    void appendMvoutRowUop(MacroCmdContext &macroCmd,
                           MpuMacroRuntime &runtime);
    void appendLoadProgressUop(MacroCmdContext &macroCmd,
                               MpuMacroRuntime &runtime);
    void appendFusedComputeProgressUop(MacroCmdContext &macroCmd,
                                       MpuMacroRuntime &runtime);

    void transitionABufferToFull(ABBufferSlot &slot, const ParsedCmd &cmd);
    void transitionABufferToLoaded(ABBufferSlot &slot, const ParsedCmd &cmd);
    void transitionCBufferToFull(CBufferSlot &slot, const ParsedCmd &cmd);
    void clearLoadedContext(BufferKind kind);
    void performCompute(const ParsedCmd &cmd);
    void performComputeFused(const ParsedCmd &cmd);
    void performDrain(const ParsedCmd &cmd);
    void releaseConsumedInputBuffers();
    std::vector<uint8_t> serializeCRow(const CBufferSlot &slot,
                                       uint32_t row) const;
    std::vector<uint8_t> serializeCTile(const CBufferSlot &slot) const;

  protected:
    MacroCmdKind classifyMacroCmd(
        const std::vector<uint8_t> &cmd) const override;
    uint32_t classifyIssueQueue(const std::vector<uint8_t> &cmd,
                                MacroCmdKind kind) const override;
    std::vector<IssueQueueState> buildIssueQueues() const override;
    bool canActivateMacroCmd(const MacroCmdContext &macroCmd) const override;
    void onMacroCmdBegin(MacroCmdContext &macroCmd) override;
    void buildUops(MacroCmdContext &macroCmd) override;
    void onMemUopComplete(MacroCmdContext &macroCmd,
                          const MemTxnContext &txn, PacketPtr pkt) override;
    void onExecUopComplete(MacroCmdContext &macroCmd,
                           const MicroOpContext &uop) override;
    void onMacroCmdEnd(MacroCmdContext &macroCmd) override;

  public:
    MpuUnit(const MpuUnitParams &params);

    uint64_t macroFifoOccupancy() const;
    uint64_t memUopQueueOccupancy() const;
    uint64_t execUopQueueOccupancy() const;
    uint64_t drainUopQueueOccupancy() const;
    uint64_t mvinCmdCount() const;
    uint64_t loadCmdCount() const;
    uint64_t computeCmdCount() const;
    uint64_t drainCmdCount() const;
    uint64_t mvoutCmdCount() const;
    uint64_t totalABytesIn() const;
    uint64_t totalBBytesIn() const;
    uint64_t totalCBytesOut() const;
    uint64_t totalOutputElementsDrained() const;
    uint64_t totalMacOps() const;
    uint64_t busyCycles() const;
    uint64_t idleCycles() const;
    uint64_t stallCyclesWaitingForSpm() const;
    uint64_t stallCyclesBufferHazard() const;
    uint64_t stallCyclesOutputStorageUnavailable() const;
    uint64_t stallCyclesDrainDestBusy() const;
    uint64_t lastComputeLatencyCycles() const;
    uint64_t lastDrainLatencyCycles() const;
    uint64_t lastOutputReadyLatencyCycles() const;
    uint64_t lastCommandLatencyCycles() const;
    int currentCmdKind() const;
    int aBufferState(uint32_t index) const;
    int bBufferState(uint32_t index) const;
    int cBufferState(uint32_t index) const;
    int outputStorageStateCode() const;
    int loadedAIndex() const;
    int loadedBIndex() const;
    bool scoreboardLoadedAReady() const;
    bool scoreboardLoadedBReady() const;
    bool scoreboardOutputReady() const;
};

} // namespace gem5

#endif // __NPU_MPU_UNIT_HH__
