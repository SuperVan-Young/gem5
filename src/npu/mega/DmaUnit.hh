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
#include <vector>

#include "npu/mega/SpecializedExecutionUnit.hh"
#include "params/DmaUnit.hh"

namespace gem5
{

class DmaUnit : public SpecializedExecutionUnit
{
  private:
    enum class XferMode : uint8_t
    {
        DramToSpm = 0,
        SpmToDram = 1,
        SpmToSpm = 2,
        DramToDram = 3,
    };

    enum class MemorySpace : uint8_t
    {
        Dram,
        Spm,
    };

    enum class RequestKind : uint8_t
    {
        None,
        GatherRead,
        ScatterDestRead,
        ScatterWrite,
        CompletionSyncWrite,
    };

    struct ParsedCmd
    {
        uint8_t deviceId = 0;
        uint8_t dataType = 0;
        uint8_t xferMode = 0;
        uint8_t syncIndicator = 0;
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

    struct BatchPlan
    {
        uint32_t startY = 0;
        uint32_t startX = 0;
        uint32_t height = 0;
        uint32_t width = 0;
        std::vector<uint8_t> buffer;
        std::vector<SourceLine> sourceLines;
        std::vector<DestLine> destLines;
    };

    static constexpr size_t CacheLineBytes = 64;
    static constexpr uint8_t DmaDeviceType = 0x4;
    static constexpr uint8_t SyncSetOpCode = 0x1;
    static constexpr size_t MaxBufferBytes = 256 * 1024 * 1024ULL;

    const size_t bufferSize;

    ParsedCmd parsedCmd;
    BatchPlan batchPlan;
    RequestKind requestKind;
    bool parsedCmdValid;
    uint32_t currentY;
    uint32_t currentX;
    size_t gatherIndex;
    size_t scatterIndex;

    uint32_t extractWord(const std::vector<uint8_t> &cmd, size_t index) const;
    ParsedCmd parseCommand(const std::vector<uint8_t> &cmd) const;
    void validateParsedCommand(const ParsedCmd &cmd) const;
    MemorySpace sourceSpace() const;
    MemorySpace destSpace() const;
    bool spaceContains(MemorySpace space, Addr addr, size_t size) const;
    void validateBaseAddress(Addr addr, MemorySpace space,
                             const char *label) const;
    void validateBurstLine(Addr addr, MemorySpace space,
                           const char *label) const;
    Addr computeTensorAddr(Addr base, uint32_t strideH, uint32_t strideW,
                           uint32_t strideC, uint16_t k, uint32_t channels,
                           uint32_t y, uint32_t x, uint32_t z) const;
    void resetCommandState();
    bool done() const;
    void advanceBatchCursor();
    void planCurrentBatch();
    void buildBatchLines();
    void issueNextGatherRead();
    void issueNextScatterRead();
    void issueScatterWrite();
    PacketPtr makeReadPacket(Addr addr) const;
    PacketPtr makeWritePacket(Addr addr, const uint8_t *data) const;
    void handleGatherReadResponse(PacketPtr pkt);
    void handleScatterReadResponse(PacketPtr pkt);
    void handleScatterWriteResponse(PacketPtr pkt);
    void finishCurrentBatch();

  protected:
    void startExecuteCommand(const std::vector<uint8_t> &cmd) override;
    bool handleMemResponse(PacketPtr pkt) override;
    bool buildCompletionSyncWord(const std::vector<uint8_t> &cmd,
                                 uint32_t &word) const override;
    void sendCompletionSyncWord(uint32_t word) override;

  public:
    DmaUnit(const DmaUnitParams &params);
};

} // namespace gem5

#endif // __NPU_MEGA_DMA_UNIT_HH__
