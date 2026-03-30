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

#ifndef __NPU_MEGA_DMA_UNIT_HH__
#define __NPU_MEGA_DMA_UNIT_HH__

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "npu/mega/SpecializedExecutionUnit.hh"
#include "params/DmaUnit.hh"

namespace gem5
{

class DmaUnit : public SpecializedExecutionUnit
{
  private:
    enum class Mode : uint8_t
    {
        MoveLayout = 0,
        Transpose = 1,
        Fill = 2,
    };

    enum class CutDim : uint8_t
    {
        H = 0,
        W = 1,
        C = 2,
        Reserved = 3,
    };

    enum class MemorySpace : uint8_t
    {
        Dram = 0,
        Spm = 1,
        DmaBank = 2,
        Invalid = 3,
    };

    enum class PendingMvinKind : uint8_t
    {
        SourceLine,
        DestLine,
    };

    struct ParsedCmd
    {
        uint8_t deviceId = 0;
        uint8_t dataType = 0;
        uint8_t mode = 0;
        uint8_t syncIndicator = 0;
        MemorySpace srcMemSpace = MemorySpace::Dram;
        MemorySpace dstMemSpace = MemorySpace::Dram;
        uint8_t srcCutDim = 0;
        uint8_t dstCutDim = 0;
        uint8_t transposeDimA = 0;
        uint8_t transposeDimB = 0;
        uint8_t srcBankId = 0;
        uint8_t dstBankId = 0;
        uint32_t modeCfg = 0;
        uint32_t bankCfg = 0;
        uint32_t word15 = 0;
        uint8_t fillValue = 0;
        Addr srcBaseAddr = 0;
        Addr dstBaseAddr = 0;
        uint32_t shapeH = 0;
        uint32_t shapeW = 0;
        uint32_t shapeC = 0;
        uint32_t srcStrideH = 0;
        uint32_t srcStrideW = 0;
        uint32_t srcStrideC = 0;
        uint32_t dstStrideH = 0;
        uint32_t dstStrideW = 0;
        uint32_t dstStrideC = 0;
        uint16_t dstK = 0;
        uint16_t srcK = 0;
    };

    struct SourceCopy
    {
        size_t bufferOffset = 0;
        uint8_t lineOffset = 0;
    };

    struct SourceLine
    {
        Addr lineAddr = 0;
        std::vector<SourceCopy> copies;
    };

    struct DestCopy
    {
        uint8_t lineOffset = 0;
        size_t bufferOffset = 0;
    };

    struct DestLine
    {
        Addr lineAddr = 0;
        std::vector<DestCopy> copies;
        std::array<uint8_t, 64> lineData = {};
    };

    struct IterationPlan
    {
        uint32_t startY = 0;
        uint32_t startX = 0;
        uint32_t height = 0;
        uint32_t width = 0;
        uint8_t transposeRemainingDim = 0;
        uint32_t transposeRemainingIndex = 0;
        Tick execLatency = 0;
        std::vector<uint8_t> sourceBuffer;
        std::vector<uint8_t> buffer;
        std::vector<SourceLine> sourceLines;
        std::vector<DestLine> destLines;
    };

    struct PendingMvinTxn
    {
        uint64_t iteration = 0;
        PendingMvinKind kind = PendingMvinKind::SourceLine;
        size_t index = 0;
    };

    static constexpr size_t CacheLineBytes = 64;
    static constexpr uint8_t DmaDeviceType = 0x4;
    static constexpr size_t MaxBankBytes = 16 * 1024 * 1024ULL;
    static constexpr size_t MaxNumBanks = 16;

    const size_t numBanks;
    const size_t bankSize;
    const Tick transposeUnitLatency;

    ParsedCmd parsedCmd;
    bool parsedCmdValid;
    std::vector<IterationPlan> iterationPlans;
    std::unordered_map<uint64_t, PendingMvinTxn> pendingMvinTxns;

    uint32_t extractWord(const std::vector<uint8_t> &cmd, size_t index) const;
    ParsedCmd parseCommand(const std::vector<uint8_t> &cmd) const;
    void validateParsedCommand(const ParsedCmd &cmd) const;
    void validateMoveLayoutCommand(const ParsedCmd &cmd) const;
    void validateTransposeCommand(const ParsedCmd &cmd) const;
    void validateFillCommand(const ParsedCmd &cmd) const;
    size_t fillRequiredBytes(const ParsedCmd &cmd) const;
    size_t transposeRequiredBytes(const ParsedCmd &cmd) const;
    uint32_t axisExtent(const ParsedCmd &cmd, uint8_t dim) const;
    bool isExternalSpace(MemorySpace space) const;
    std::vector<Addr> externalFillLineAddrs(const ParsedCmd &cmd) const;
    MemorySpace sourceSpace() const;
    MemorySpace destSpace() const;
    bool spaceContains(MemorySpace space, Addr addr, size_t size) const;
    void validateBaseAddress(Addr addr, MemorySpace space,
                             const char *label) const;
    void validateBurstLine(Addr addr, MemorySpace space,
                           const char *label) const;
    Addr computeTensorAddr(Addr base, uint32_t strideH, uint32_t strideW,
                           uint32_t strideC, uint16_t k, uint32_t width,
                           uint32_t channels, uint8_t cutDim,
                           uint32_t y, uint32_t x, uint32_t z) const;
    void resetCommandState();
    IterationPlan &iterationPlan(uint64_t iteration);
    const IterationPlan *findIterationPlan(uint64_t iteration) const;
    void buildIterationPlans(ActiveExecution &exec);
    void buildMoveLayoutPlans();
    void buildTransposePlans();
    void buildFillPlans();
    void buildBatchLines(IterationPlan &plan) const;
    void buildTransposeLines(IterationPlan &plan) const;

  protected:
    void startExecuteCommand(const std::vector<uint8_t> &cmd) override;
    void onCommandBegin(ActiveExecution &exec) override;
    void prologue(ActiveExecution &exec) override;
    void buildMvinRequests(ActiveExecution &exec,
                           std::vector<MemRequestDesc> &reqs) override;
    void onMvinResponse(ActiveExecution &exec,
                        const MemTxnContext &txn,
                        PacketPtr pkt) override;
    Tick execute(ActiveExecution &exec) override;
    void buildMvoutRequests(ActiveExecution &exec,
                            std::vector<MemRequestDesc> &reqs) override;

  public:
    DmaUnit(const DmaUnitParams &params);
};

} // namespace gem5

#endif // __NPU_MEGA_DMA_UNIT_HH__
