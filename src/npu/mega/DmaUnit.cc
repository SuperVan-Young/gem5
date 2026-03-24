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

#include "npu/mega/DmaUnit.hh"

#include <algorithm>
#include <cstring>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/DmaUnit.hh"

namespace gem5
{

namespace
{

constexpr Addr DramBase = 0x20000000ULL;
constexpr Addr DramEnd = 0x5fffffffULL;
constexpr Addr SpmBase = 0x60000000ULL;
constexpr Addr SpmEnd = 0x6fffffffULL;

} // namespace

DmaUnit::DmaUnit(const DmaUnitParams &params)
    : SpecializedExecutionUnit(params),
      bufferSize(params.buffer_size),
      parsedCmdValid(false),
      currentY(0),
      currentX(0)
{
    fatal_if(macroCmdBytes != CacheLineBytes,
             "%s: DmaUnit requires 64-byte commands", name());
    fatal_if(bufferSize == 0 || bufferSize > MaxBufferBytes,
             "%s: DmaUnit buffer_size must be in the range [1, %zu]",
             name(), MaxBufferBytes);
    fatal_if(memSidePorts.size() != 1,
             "%s: DmaUnit currently supports exactly one mem_side port",
             name());
}

uint32_t
DmaUnit::extractWord(const std::vector<uint8_t> &cmd, size_t index) const
{
    panic_if((index + 1) * sizeof(uint32_t) > cmd.size(),
             "DmaUnit: command word %zu is out of range", index);

    uint32_t word = 0;
    std::memcpy(&word, cmd.data() + (index * sizeof(uint32_t)), sizeof(word));
    return word;
}

DmaUnit::ParsedCmd
DmaUnit::parseCommand(const std::vector<uint8_t> &cmd) const
{
    panic_if(cmd.size() != CacheLineBytes,
             "DmaUnit: expected 64-byte command, got %zu bytes", cmd.size());

    ParsedCmd parsed;
    const uint32_t header = extractWord(cmd, 0);
    const uint8_t opCode = (header >> 16) & 0xff;

    parsed.deviceId = (header >> 24) & 0xf;
    parsed.dataType = (opCode >> 5) & 0x7;
    parsed.xferMode = (opCode >> 2) & 0x7;
    parsed.syncIndicator = (header >> 8) & 0xff;
    parsed.srcBaseAddr = extractWord(cmd, 1);
    parsed.dstBaseAddr = extractWord(cmd, 2);
    parsed.shapeH = extractWord(cmd, 3);
    parsed.shapeW = extractWord(cmd, 4);
    parsed.shapeC = extractWord(cmd, 5);
    parsed.srcStrideH = extractWord(cmd, 6);
    parsed.srcStrideW = extractWord(cmd, 7);
    parsed.srcStrideC = extractWord(cmd, 8);
    parsed.dstStrideH = extractWord(cmd, 9);
    parsed.dstStrideW = extractWord(cmd, 10);
    parsed.dstStrideC = extractWord(cmd, 11);

    const uint32_t blockCfg = extractWord(cmd, 12);
    parsed.dstK = (blockCfg >> 16) & 0xffff;
    parsed.srcK = blockCfg & 0xffff;

    panic_if(((header >> 28) & 0xf) != DmaDeviceType,
             "DmaUnit: unexpected device_type=%u", (header >> 28) & 0xf);

    return parsed;
}

void
DmaUnit::validateParsedCommand(const ParsedCmd &cmd) const
{
    panic_if(cmd.dataType != 0,
             "DmaUnit: unsupported data_type=%u", cmd.dataType);
    panic_if(cmd.xferMode > static_cast<uint8_t>(XferMode::DramToDram),
             "DmaUnit: unsupported xfer_mode=%u", cmd.xferMode);

    validateBaseAddress(cmd.srcBaseAddr, sourceSpace(), "source");
    validateBaseAddress(cmd.dstBaseAddr, destSpace(), "destination");

    if (cmd.srcK > 0) {
        panic_if(cmd.shapeW % cmd.srcK != 0,
                 "DmaUnit: source blocked layout requires W %% k == 0");
    }
    if (cmd.dstK > 0) {
        panic_if(cmd.shapeW % cmd.dstK != 0,
                 "DmaUnit: destination blocked layout requires W %% k == 0");
    }
}

DmaUnit::MemorySpace
DmaUnit::sourceSpace() const
{
    switch (static_cast<XferMode>(parsedCmd.xferMode)) {
      case XferMode::DramToSpm:
      case XferMode::DramToDram:
        return MemorySpace::Dram;
      case XferMode::SpmToDram:
      case XferMode::SpmToSpm:
        return MemorySpace::Spm;
    }

    panic("DmaUnit: unreachable source xfer mode");
}

DmaUnit::MemorySpace
DmaUnit::destSpace() const
{
    switch (static_cast<XferMode>(parsedCmd.xferMode)) {
      case XferMode::DramToSpm:
      case XferMode::SpmToSpm:
        return MemorySpace::Spm;
      case XferMode::SpmToDram:
      case XferMode::DramToDram:
        return MemorySpace::Dram;
    }

    panic("DmaUnit: unreachable destination xfer mode");
}

bool
DmaUnit::spaceContains(MemorySpace space, Addr addr, size_t size) const
{
    panic_if(size == 0, "DmaUnit: zero-sized memory validation is invalid");

    const Addr base = (space == MemorySpace::Dram) ? DramBase : SpmBase;
    const Addr end = (space == MemorySpace::Dram) ? DramEnd : SpmEnd;
    if (addr < base || addr > end) {
        return false;
    }

    const Addr size_minus_one = size - 1;
    if (addr > end - size_minus_one) {
        return false;
    }

    return true;
}

void
DmaUnit::validateBaseAddress(
    Addr addr, MemorySpace space, const char *label) const
{
    panic_if(!spaceContains(space, addr, 1),
             "DmaUnit: invalid %s base address %#llx", label,
             static_cast<unsigned long long>(addr));
}

void
DmaUnit::validateBurstLine(
    Addr addr, MemorySpace space, const char *label) const
{
    panic_if(addr % CacheLineBytes != 0,
             "DmaUnit: %s burst start address %#llx is not 64B aligned",
             label, static_cast<unsigned long long>(addr));
    panic_if(!spaceContains(space, addr, CacheLineBytes),
             "DmaUnit: %s burst address %#llx crosses invalid region",
             label, static_cast<unsigned long long>(addr));
}

Addr
DmaUnit::computeTensorAddr(Addr base, uint32_t strideH, uint32_t strideW,
                           uint32_t strideC, uint16_t k, uint32_t channels,
                           uint32_t y, uint32_t x, uint32_t z) const
{
    if (k == 0) {
        return base + static_cast<Addr>(y) * strideH +
               static_cast<Addr>(x) * strideW +
               static_cast<Addr>(z) * strideC;
    }

    return base + static_cast<Addr>(y) * strideH +
           static_cast<Addr>(x / k) * static_cast<Addr>(strideC) * channels +
           static_cast<Addr>(z) * strideC +
           static_cast<Addr>(x % k) * strideW;
}

void
DmaUnit::resetCommandState()
{
    parsedCmd = ParsedCmd();
    batchPlan = BatchPlan();
    parsedCmdValid = false;
    currentY = 0;
    currentX = 0;
    pendingMvinTxns.clear();
}

bool
DmaUnit::done() const
{
    return !parsedCmdValid || parsedCmd.shapeH == 0 || parsedCmd.shapeW == 0 ||
           parsedCmd.shapeC == 0 || currentY >= parsedCmd.shapeH;
}

void
DmaUnit::advanceBatchCursor()
{
    if (batchPlan.width == parsedCmd.shapeW && batchPlan.startX == 0) {
        currentY += batchPlan.height;
        currentX = 0;
        return;
    }

    currentX += batchPlan.width;
    if (currentX >= parsedCmd.shapeW) {
        currentX = 0;
        currentY += 1;
    }
}

void
DmaUnit::planCurrentBatch()
{
    batchPlan = BatchPlan();
    batchPlan.startY = currentY;
    batchPlan.startX = currentX;

    if (done()) {
        return;
    }

    const size_t channels = parsedCmd.shapeC;
    panic_if(channels > bufferSize,
             "DmaUnit: buffer_size=%zu is too small for a (1,1,C) tile",
             bufferSize);

    const uint32_t remainingH = parsedCmd.shapeH - currentY;
    const uint32_t remainingW = parsedCmd.shapeW - currentX;

    if (currentX == 0) {
        const size_t hSliceBytes =
            static_cast<size_t>(parsedCmd.shapeW) * parsedCmd.shapeC;
        if (hSliceBytes <= bufferSize) {
            batchPlan.height = std::max<uint32_t>(
                1, std::min<uint32_t>(remainingH, bufferSize / hSliceBytes));
            batchPlan.width = parsedCmd.shapeW;
        } else {
            batchPlan.height = 1;
            batchPlan.width = std::max<uint32_t>(
                1, std::min<uint32_t>(remainingW, bufferSize / channels));
        }
    } else {
        batchPlan.height = 1;
        batchPlan.width = std::max<uint32_t>(
            1, std::min<uint32_t>(remainingW, bufferSize / channels));
    }

    panic_if(batchPlan.width == 0,
             "DmaUnit: failed to plan a non-empty batch");

    const size_t batchBytes = static_cast<size_t>(batchPlan.height) *
                              batchPlan.width * parsedCmd.shapeC;
    batchPlan.buffer.assign(batchBytes, 0);
    buildBatchLines();

    DPRINTF(DmaUnit,
            "Planned batch y=%u x=%u h=%u w=%u src_lines=%zu dst_lines=%zu\n",
            batchPlan.startY, batchPlan.startX, batchPlan.height,
            batchPlan.width, batchPlan.sourceLines.size(),
            batchPlan.destLines.size());
}

void
DmaUnit::buildBatchLines()
{
    std::map<Addr, std::vector<SourceCopy>> sourceMap;
    std::map<Addr, std::vector<DestCopy>> destMap;

    for (uint32_t localY = 0; localY < batchPlan.height; ++localY) {
        for (uint32_t localX = 0; localX < batchPlan.width; ++localX) {
            for (uint32_t z = 0; z < parsedCmd.shapeC; ++z) {
                const uint32_t globalY = batchPlan.startY + localY;
                const uint32_t globalX = batchPlan.startX + localX;
                const size_t bufferOffset =
                    (static_cast<size_t>(localY) * batchPlan.width + localX) *
                        parsedCmd.shapeC +
                    z;

                const Addr srcAddr = computeTensorAddr(
                    parsedCmd.srcBaseAddr, parsedCmd.srcStrideH,
                    parsedCmd.srcStrideW, parsedCmd.srcStrideC, parsedCmd.srcK,
                    parsedCmd.shapeC, globalY, globalX, z);
                const Addr srcLineAddr = srcAddr & ~(CacheLineBytes - 1);
                validateBurstLine(srcLineAddr, sourceSpace(), "source");
                sourceMap[srcLineAddr].push_back({
                    bufferOffset,
                    static_cast<uint8_t>(srcAddr - srcLineAddr),
                });

                const Addr dstAddr = computeTensorAddr(
                    parsedCmd.dstBaseAddr, parsedCmd.dstStrideH,
                    parsedCmd.dstStrideW, parsedCmd.dstStrideC, parsedCmd.dstK,
                    parsedCmd.shapeC, globalY, globalX, z);
                const Addr dstLineAddr = dstAddr & ~(CacheLineBytes - 1);
                validateBurstLine(dstLineAddr, destSpace(), "destination");
                destMap[dstLineAddr].push_back({
                    static_cast<uint8_t>(dstAddr - dstLineAddr),
                    bufferOffset,
                });
            }
        }
    }

    for (const auto &entry : sourceMap) {
        batchPlan.sourceLines.push_back({entry.first, entry.second});
    }
    for (const auto &entry : destMap) {
        batchPlan.destLines.push_back({entry.first, entry.second, {}});
    }
}

void
DmaUnit::startExecuteCommand(const std::vector<uint8_t> &cmd)
{
    const uint64_t totalPrologues = activeExecution.prologueCount;
    const uint64_t totalExecutes = activeExecution.executeCount;
    const uint64_t totalEpilogues = activeExecution.epilogueCount;
    const uint64_t totalReads = activeExecution.completedReadRespCount;
    const uint64_t totalWrites = activeExecution.completedWriteRespCount;
    const uint64_t totalIterations = activeExecution.completedIterations;

    activeExecution = ActiveExecution{};
    activeExecution.cmd = cmd;
    activeExecution.fields = parseCmdFields(extractCmdWord(cmd));
    activeExecution.phase = Phase::Prologue;
    activeExecution.prologueCount = totalPrologues;
    activeExecution.executeCount = totalExecutes;
    activeExecution.epilogueCount = totalEpilogues;
    activeExecution.completedReadRespCount = totalReads;
    activeExecution.completedWriteRespCount = totalWrites;
    activeExecution.completedIterations = totalIterations;
    activeExecution.readMask = 0;
    activeExecution.writeMask = 0;
    activeExecution.repetition = 1;
    activeExecution.reserved = 0;

    onCommandBegin(activeExecution);
    advanceActivePhase(activeExecution);
}

void
DmaUnit::onCommandBegin(ActiveExecution &exec)
{
    (void)exec;
    resetCommandState();
    parsedCmd = parseCommand(exec.cmd);
    parsedCmdValid = true;
    validateParsedCommand(parsedCmd);
}

void
DmaUnit::prologue(ActiveExecution &exec)
{
    (void)exec;
    planCurrentBatch();
}

void
DmaUnit::buildMvinRequests(ActiveExecution &exec,
                           std::vector<MemRequestDesc> &reqs)
{
    (void)exec;
    pendingMvinTxns.clear();

    if (done()) {
        return;
    }

    uint64_t token = nextMemTxnToken;
    for (size_t i = 0; i < batchPlan.sourceLines.size(); ++i) {
        const auto &line = batchPlan.sourceLines[i];
        MemRequestDesc req;
        req.portId = 0;
        req.kind = MemTxnContext::Kind::Mvin;
        req.addr = line.lineAddr;
        req.size = CacheLineBytes;
        reqs.push_back(req);
        pendingMvinTxns.emplace(token++, PendingMvinTxn{
            PendingMvinKind::SourceLine, i});
    }

    for (size_t i = 0; i < batchPlan.destLines.size(); ++i) {
        const auto &line = batchPlan.destLines[i];
        MemRequestDesc req;
        req.portId = 0;
        req.kind = MemTxnContext::Kind::Mvin;
        req.addr = line.lineAddr;
        req.size = CacheLineBytes;
        reqs.push_back(req);
        pendingMvinTxns.emplace(token++, PendingMvinTxn{
            PendingMvinKind::DestLine, i});
    }
}

void
DmaUnit::onMvinResponse(ActiveExecution &exec,
                        const MemTxnContext &txn,
                        PacketPtr pkt)
{
    (void)exec;
    auto it = pendingMvinTxns.find(txn.token);
    panic_if(it == pendingMvinTxns.end(),
             "%s: unexpected DMA mvin token=%llu", name(),
             static_cast<unsigned long long>(txn.token));

    const PendingMvinTxn pending = it->second;
    pendingMvinTxns.erase(it);

    switch (pending.kind) {
      case PendingMvinKind::SourceLine: {
        const auto &line = batchPlan.sourceLines.at(pending.index);
        const uint8_t *data = pkt->getConstPtr<uint8_t>();
        for (const auto &copy : line.copies) {
            batchPlan.buffer[copy.bufferOffset] = data[copy.lineOffset];
        }
        break;
      }
      case PendingMvinKind::DestLine: {
        auto &line = batchPlan.destLines.at(pending.index);
        std::memcpy(line.lineData.data(), pkt->getConstPtr<uint8_t>(),
                    CacheLineBytes);
        break;
      }
    }
}

Tick
DmaUnit::execute(ActiveExecution &exec)
{
    (void)exec;

    if (done()) {
        return 0;
    }

    for (auto &line : batchPlan.destLines) {
        for (const auto &copy : line.copies) {
            line.lineData[copy.lineOffset] =
                batchPlan.buffer[copy.bufferOffset];
        }
    }

    return 0;
}

void
DmaUnit::buildMvoutRequests(ActiveExecution &exec,
                            std::vector<MemRequestDesc> &reqs)
{
    (void)exec;

    if (done()) {
        return;
    }

    for (const auto &line : batchPlan.destLines) {
        MemRequestDesc req;
        req.portId = 0;
        req.kind = MemTxnContext::Kind::Mvout;
        req.addr = line.lineAddr;
        req.size = CacheLineBytes;
        req.data.assign(line.lineData.begin(), line.lineData.end());
        reqs.push_back(req);
    }
}

void
DmaUnit::epilogue(ActiveExecution &exec)
{
    (void)exec;

    if (!batchPlan.buffer.empty()) {
        advanceBatchCursor();
    }
}

bool
DmaUnit::shouldExit(const ActiveExecution &exec) const
{
    (void)exec;
    return done();
}

} // namespace gem5
