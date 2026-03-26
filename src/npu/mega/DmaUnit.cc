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
constexpr uint32_t MoveLayoutModeCfgMask = 0x3fU;
constexpr uint32_t TransposeModeCfgMask = 0x3fU;
constexpr uint32_t FillModeCfgMask = 0x3U;
constexpr uint32_t TransposeBankCfgMask = 0xffU;
constexpr uint32_t FillBankCfgMask = 0x0fU;

} // namespace

DmaUnit::DmaUnit(const DmaUnitParams &params)
    : SpecializedExecutionUnit(params),
      numBanks(params.num_banks),
      bankSize(params.bank_size),
      transposeUnitLatency(params.transpose_unit_latency),
      bankWorkspace(numBanks),
      parsedCmdValid(false),
      currentY(0),
      currentX(0)
{
    fatal_if(macroCmdBytes != CacheLineBytes,
             "%s: DmaUnit requires 64-byte commands", name());

    fatal_if(numBanks == 0 || numBanks > MaxNumBanks,
             "%s: DmaUnit num_banks must be in the range [1, %zu]",
             name(), MaxNumBanks);
    fatal_if(bankSize == 0 || bankSize > MaxBankBytes ||
                 (bankSize & (bankSize - 1)) != 0,
             "%s: DmaUnit bank_size must be a power of two "
             "in the range [1, %zu]",
             name(), MaxBankBytes);
    fatal_if(memSidePorts.empty() || memSidePorts.size() > 2,
             "%s: DmaUnit requires one or two mem_side ports",
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
    parsed.mode = (opCode >> 2) & 0x7;
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
    parsed.modeCfg = extractWord(cmd, 13);
    parsed.bankCfg = extractWord(cmd, 14);
    parsed.word15 = extractWord(cmd, 15);
    parsed.fillValue = parsed.word15 & 0xffU;

    if (parsed.mode <= static_cast<uint8_t>(Mode::Fill)) {
        switch (static_cast<Mode>(parsed.mode)) {
          case Mode::MoveLayout:
            parsed.srcMemSpace =
                (parsed.modeCfg & 0x1U) ? MemorySpace::Spm : MemorySpace::Dram;
            parsed.dstMemSpace =
                (parsed.modeCfg & 0x2U) ? MemorySpace::Spm : MemorySpace::Dram;
            parsed.srcCutDim = (parsed.modeCfg >> 2) & 0x3;
            parsed.dstCutDim = (parsed.modeCfg >> 4) & 0x3;
            break;
          case Mode::Transpose:
            parsed.srcMemSpace =
                (parsed.modeCfg & 0x1U) ? MemorySpace::Spm : MemorySpace::Dram;
            parsed.dstMemSpace =
                (parsed.modeCfg & 0x2U) ? MemorySpace::Spm : MemorySpace::Dram;
            parsed.transposeDimA = (parsed.modeCfg >> 2) & 0x3;
            parsed.transposeDimB = (parsed.modeCfg >> 4) & 0x3;
            parsed.srcBankId = parsed.bankCfg & 0xf;
            parsed.dstBankId = (parsed.bankCfg >> 4) & 0xf;
            break;
          case Mode::Fill:
            switch (parsed.modeCfg & FillModeCfgMask) {
              case 0x0U:
                parsed.dstMemSpace = MemorySpace::Dram;
                break;
              case 0x1U:
                parsed.dstMemSpace = MemorySpace::Spm;
                break;
              case 0x2U:
                parsed.dstMemSpace = MemorySpace::DmaBank;
                break;
              default:
                parsed.dstMemSpace = MemorySpace::Invalid;
                break;
            }
            parsed.dstBankId = parsed.bankCfg & 0xf;
            break;
        }
    }

    panic_if(((header >> 28) & 0xf) != DmaDeviceType,
             "DmaUnit: unexpected device_type=%u", (header >> 28) & 0xf);

    return parsed;
}

void
DmaUnit::validateParsedCommand(const ParsedCmd &cmd) const
{
    panic_if(cmd.dataType != 0,
             "DmaUnit: unsupported data_type=%u", cmd.dataType);
    panic_if(cmd.mode > static_cast<uint8_t>(Mode::Fill),
             "DmaUnit: unsupported mode=%u", cmd.mode);

    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::MoveLayout:
      case Mode::Transpose:
        panic_if(cmd.word15 != 0,
                 "DmaUnit: Word 15 must be zero, got %#x", cmd.word15);
        if (static_cast<Mode>(cmd.mode) == Mode::MoveLayout) {
            validateMoveLayoutCommand(cmd);
            return;
        }
        validateTransposeCommand(cmd);
        return;
      case Mode::Fill:
        validateFillCommand(cmd);
        return;
    }

    panic("DmaUnit: unreachable mode validation");
}

void
DmaUnit::validateMoveLayoutCommand(const ParsedCmd &cmd) const
{
    panic_if((cmd.modeCfg & ~MoveLayoutModeCfgMask) != 0,
             "DmaUnit: reserved mode_cfg bits set for move_layout");
    panic_if(cmd.srcCutDim > static_cast<uint8_t>(CutDim::C),
             "DmaUnit: reserved src_cut_dim=%u", cmd.srcCutDim);
    panic_if(cmd.dstCutDim > static_cast<uint8_t>(CutDim::C),
             "DmaUnit: reserved dst_cut_dim=%u", cmd.dstCutDim);
    panic_if(cmd.bankCfg != 0,
             "DmaUnit: move_layout requires bank_cfg == 0");

    validateBaseAddress(cmd.srcBaseAddr, cmd.srcMemSpace, "source");
    validateBaseAddress(cmd.dstBaseAddr, cmd.dstMemSpace, "destination");

    auto validateBlockedK = [&](uint16_t k, uint8_t cutDim,
                                const char *label) {
        if (k == 0) {
            return;
        }

        switch (static_cast<CutDim>(cutDim)) {
          case CutDim::H: {
            const bool invalidBlockedK = (cmd.shapeH % k) != 0;
            panic_if(invalidBlockedK,
                     "DmaUnit: %s blocked layout requires H %% k == 0",
                     label);
            return;
          }
          case CutDim::W: {
            const bool invalidBlockedK = (cmd.shapeW % k) != 0;
            panic_if(invalidBlockedK,
                     "DmaUnit: %s blocked layout requires W %% k == 0",
                     label);
            return;
          }
          case CutDim::C: {
            const bool invalidBlockedK = (cmd.shapeC % k) != 0;
            panic_if(invalidBlockedK,
                     "DmaUnit: %s blocked layout requires C %% k == 0",
                     label);
            return;
          }
          case CutDim::Reserved:
            break;
        }

        panic("DmaUnit: unreachable move_layout cut dimension");
    };

    validateBlockedK(cmd.srcK, cmd.srcCutDim, "source");
    validateBlockedK(cmd.dstK, cmd.dstCutDim, "destination");
}

void
DmaUnit::validateTransposeCommand(const ParsedCmd &cmd) const
{
    panic_if((cmd.modeCfg & ~TransposeModeCfgMask) != 0,
             "DmaUnit: reserved mode_cfg bits set for transpose");
    panic_if(cmd.transposeDimA > static_cast<uint8_t>(CutDim::C),
             "DmaUnit: reserved transpose_dim_a=%u", cmd.transposeDimA);
    panic_if(cmd.transposeDimB > static_cast<uint8_t>(CutDim::C),
             "DmaUnit: reserved transpose_dim_b=%u", cmd.transposeDimB);
    panic_if((cmd.bankCfg & ~TransposeBankCfgMask) != 0,
             "DmaUnit: reserved bank_cfg bits set for transpose");
    panic_if(cmd.srcBankId >= numBanks,
             "DmaUnit: src_bank_id=%u exceeds num_banks=%u",
             cmd.srcBankId, static_cast<unsigned>(numBanks));
    panic_if(cmd.dstBankId >= numBanks,
             "DmaUnit: dst_bank_id=%u exceeds num_banks=%u",
             cmd.dstBankId, static_cast<unsigned>(numBanks));
    panic_if(cmd.srcBankId == cmd.dstBankId,
             "DmaUnit: transpose requires src_bank_id != dst_bank_id");
    panic_if(cmd.transposeDimA == cmd.transposeDimB,
             "DmaUnit: transpose requires transpose_dim_a != transpose_dim_b");

    validateBaseAddress(cmd.srcBaseAddr, cmd.srcMemSpace, "source");
    validateBaseAddress(cmd.dstBaseAddr, cmd.dstMemSpace, "destination");

    panic_if(cmd.srcK != 0 || cmd.dstK != 0,
             "DmaUnit: transpose requires src_k == 0 and dst_k == 0");

    const size_t requiredBytes = transposeRequiredBytes(cmd);
    panic_if(requiredBytes > bankSize,
             "DmaUnit: transpose required_bytes=%llu exceeds bank_size=%u",
             static_cast<unsigned long long>(requiredBytes),
             static_cast<unsigned>(bankSize));
}

void
DmaUnit::validateFillCommand(const ParsedCmd &cmd) const
{
    panic_if((cmd.modeCfg & ~FillModeCfgMask) != 0,
             "DmaUnit: reserved mode_cfg bits set for fill");
    panic_if(cmd.dstMemSpace == MemorySpace::Invalid,
             "DmaUnit: reserved dst_mem_space=3 for fill");
    panic_if((cmd.bankCfg & ~FillBankCfgMask) != 0,
             "DmaUnit: reserved bank_cfg bits set for fill");
    panic_if((cmd.word15 & ~0xffU) != 0,
             "DmaUnit: fill requires Word 15[31:8] == 0");
    panic_if(cmd.srcBaseAddr != 0,
             "DmaUnit: fill requires src_base_addr == 0");

    const bool hasSourceLayoutFields =
        cmd.srcStrideH != 0 || cmd.srcStrideW != 0 || cmd.srcStrideC != 0 ||
        cmd.srcK != 0;
    panic_if(hasSourceLayoutFields,
             "DmaUnit: fill requires source layout fields == 0");
    panic_if(cmd.dstK != 0,
             "DmaUnit: fill requires dst_k == 0");

    if (cmd.dstMemSpace == MemorySpace::DmaBank) {
        panic_if(cmd.dstBankId >= numBanks,
                 "DmaUnit: dst_bank_id=%u exceeds num_banks=%u",
                 cmd.dstBankId, static_cast<unsigned>(numBanks));
        panic_if(cmd.dstBaseAddr != 0,
                 "DmaUnit: fill->DMA_BANK requires dst_base_addr == 0");

        const size_t requiredBytes = fillRequiredBytes(cmd);
        panic_if(requiredBytes > bankSize,
                 "DmaUnit: fill required_bytes=%llu exceeds bank_size=%u",
                 static_cast<unsigned long long>(requiredBytes),
                 static_cast<unsigned>(bankSize));
        return;
    }

    panic_if(cmd.dstBankId != 0,
             "DmaUnit: external fill requires dst_bank_id == 0");
    validateBaseAddress(cmd.dstBaseAddr, cmd.dstMemSpace, "destination");
    externalFillLineAddrs(cmd);
}

size_t
DmaUnit::fillRequiredBytes(const ParsedCmd &cmd) const
{
    if (cmd.shapeH == 0 || cmd.shapeW == 0 || cmd.shapeC == 0) {
        return 0;
    }

    const unsigned long long requiredBytes =
        static_cast<unsigned long long>(cmd.shapeH - 1) * cmd.dstStrideH +
        static_cast<unsigned long long>(cmd.shapeW - 1) * cmd.dstStrideW +
        static_cast<unsigned long long>(cmd.shapeC - 1) * cmd.dstStrideC +
        1ULL;
    return static_cast<size_t>(requiredBytes);
}

size_t
DmaUnit::transposeRequiredBytes(const ParsedCmd &cmd) const
{
    return static_cast<size_t>(cmd.shapeH) * cmd.shapeW * cmd.shapeC;
}

uint32_t
DmaUnit::axisExtent(const ParsedCmd &cmd, uint8_t dim) const
{
    switch (static_cast<CutDim>(dim)) {
      case CutDim::H:
        return cmd.shapeH;
      case CutDim::W:
        return cmd.shapeW;
      case CutDim::C:
        return cmd.shapeC;
      case CutDim::Reserved:
        break;
    }

    panic("DmaUnit: unreachable axis extent dimension");
}

bool
DmaUnit::isExternalSpace(MemorySpace space) const
{
    return space == MemorySpace::Dram || space == MemorySpace::Spm;
}

std::vector<Addr>
DmaUnit::externalFillLineAddrs(const ParsedCmd &cmd) const
{
    panic_if(!isExternalSpace(cmd.dstMemSpace),
             "DmaUnit: external fill requires DRAM or SPM destination space");

    std::map<Addr, std::array<bool, CacheLineBytes>> coverageMap;
    if (cmd.shapeH == 0 || cmd.shapeW == 0 || cmd.shapeC == 0) {
        return {};
    }

    for (uint32_t y = 0; y < cmd.shapeH; ++y) {
        for (uint32_t x = 0; x < cmd.shapeW; ++x) {
            for (uint32_t z = 0; z < cmd.shapeC; ++z) {
                const Addr dstAddr = computeTensorAddr(
                    cmd.dstBaseAddr, cmd.dstStrideH, cmd.dstStrideW,
                    cmd.dstStrideC, 0, cmd.shapeW, cmd.shapeC,
                    static_cast<uint8_t>(CutDim::W), y, x, z);
                const Addr lineAddr = dstAddr & ~(CacheLineBytes - 1);
                validateBurstLine(lineAddr, cmd.dstMemSpace,
                                  "fill destination");
                coverageMap[lineAddr][dstAddr - lineAddr] = true;
            }
        }
    }

    std::vector<Addr> lineAddrs;
    for (const auto &entry : coverageMap) {
        const bool fullyCovered = std::all_of(
            entry.second.begin(), entry.second.end(),
            [](bool covered) { return covered; });
        panic_if(!fullyCovered,
                 "DmaUnit: external fill requires full 64B cache-line "
                 "coverage at %#llx",
                 static_cast<unsigned long long>(entry.first));
        lineAddrs.push_back(entry.first);
    }

    return lineAddrs;
}

DmaUnit::MemorySpace
DmaUnit::sourceSpace() const
{
    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::MoveLayout:
      case Mode::Transpose:
        return parsedCmd.srcMemSpace;
      case Mode::Fill:
        panic("DmaUnit: fill mode has no external source space");
    }

    panic("DmaUnit: unreachable source mode");
}

DmaUnit::MemorySpace
DmaUnit::destSpace() const
{
    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::MoveLayout:
      case Mode::Transpose:
        return parsedCmd.dstMemSpace;
      case Mode::Fill:
        panic("DmaUnit: fill mode has no external destination space");
    }

    panic("DmaUnit: unreachable destination mode");
}

bool
DmaUnit::spaceContains(MemorySpace space, Addr addr, size_t size) const
{
    panic_if(!isExternalSpace(space),
             "DmaUnit: invalid external memory space validation");
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
                           uint32_t strideC, uint16_t k, uint32_t width,
                           uint32_t channels, uint8_t cutDim,
                           uint32_t y, uint32_t x, uint32_t z) const
{
    const auto linearAddr = [&]() {
        return base + static_cast<Addr>(y) * strideH +
               static_cast<Addr>(x) * strideW +
               static_cast<Addr>(z) * strideC;
    };

    if (k == 0) {
        return linearAddr();
    }

    switch (static_cast<CutDim>(cutDim)) {
      case CutDim::H:
        return base +
               static_cast<Addr>(y / k) * static_cast<Addr>(strideW) * width +
               static_cast<Addr>(x) * strideW +
               static_cast<Addr>(z) * strideC +
               static_cast<Addr>(y % k) * strideH;
      case CutDim::W:
        return base + static_cast<Addr>(y) * strideH +
               static_cast<Addr>(x / k) * static_cast<Addr>(strideC) *
                   channels +
               static_cast<Addr>(z) * strideC +
               static_cast<Addr>(x % k) * strideW;
      case CutDim::C:
        return linearAddr();
      case CutDim::Reserved:
        break;
    }

    panic("DmaUnit: unreachable tensor cut dimension");
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

    for (auto &bank : bankWorkspace) {
        if (!bank.empty()) {
            std::fill(bank.begin(), bank.end(), 0);
        }
    }
}

bool
DmaUnit::done() const
{
    if (!parsedCmdValid) {
        return true;
    }

    const Mode mode = static_cast<Mode>(parsedCmd.mode);
    if (mode == Mode::Fill || mode == Mode::Transpose) {
        return currentY != 0;
    }

    return parsedCmd.shapeH == 0 || parsedCmd.shapeW == 0 ||
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

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Fill) {
        if (parsedCmd.dstMemSpace == MemorySpace::DmaBank) {
            return;
        }

        for (Addr lineAddr : externalFillLineAddrs(parsedCmd)) {
            DestLine line;
            line.lineAddr = lineAddr;
            line.lineData.fill(parsedCmd.fillValue);
            batchPlan.destLines.push_back(line);
        }
        return;
    }

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Transpose) {
        batchPlan.buffer.assign(transposeRequiredBytes(parsedCmd), 0);
        bankWorkspace.at(parsedCmd.srcBankId).assign(bankSize, 0);
        bankWorkspace.at(parsedCmd.dstBankId).assign(bankSize, 0);
        buildTransposeLines();
        return;
    }

    const size_t channels = parsedCmd.shapeC;
    panic_if(channels > bankSize,
             "DmaUnit: bank_size=%zu is too small for a (1,1,C) tile",
             bankSize);

    const uint32_t remainingH = parsedCmd.shapeH - currentY;
    const uint32_t remainingW = parsedCmd.shapeW - currentX;

    if (currentX == 0) {
        const size_t hSliceBytes =
            static_cast<size_t>(parsedCmd.shapeW) * parsedCmd.shapeC;
        if (hSliceBytes <= bankSize) {
            batchPlan.height = std::max<uint32_t>(
                1, std::min<uint32_t>(remainingH, bankSize / hSliceBytes));
            batchPlan.width = parsedCmd.shapeW;
        } else {
            batchPlan.height = 1;
            batchPlan.width = std::max<uint32_t>(
                1, std::min<uint32_t>(remainingW, bankSize / channels));
        }
    } else {
        batchPlan.height = 1;
        batchPlan.width = std::max<uint32_t>(
            1, std::min<uint32_t>(remainingW, bankSize / channels));
    }

    panic_if(batchPlan.width == 0,
             "DmaUnit: failed to plan a non-empty batch");

    const size_t batchBytes = static_cast<size_t>(batchPlan.height) *
                              batchPlan.width * parsedCmd.shapeC;
    panic_if(batchBytes > bankSize,
             "DmaUnit: planned move_layout batch requires %zu bytes, "
             "exceeds bank_size=%zu",
             batchBytes, bankSize);
    bankWorkspace.front().assign(bankSize, 0);
    buildBatchLines();

    DPRINTF(DmaUnit,
            "Planned batch y=%u x=%u h=%u w=%u src_lines=%u dst_lines=%u\n",
            batchPlan.startY, batchPlan.startX, batchPlan.height,
            batchPlan.width,
            static_cast<unsigned>(batchPlan.sourceLines.size()),
            static_cast<unsigned>(batchPlan.destLines.size()));
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
                    parsedCmd.shapeW, parsedCmd.shapeC, parsedCmd.srcCutDim,
                    globalY, globalX, z);
                const Addr srcLineAddr = srcAddr & ~(CacheLineBytes - 1);
                validateBurstLine(srcLineAddr, sourceSpace(), "source");
                sourceMap[srcLineAddr].push_back({
                    bufferOffset,
                    static_cast<uint8_t>(srcAddr - srcLineAddr),
                });

                const Addr dstAddr = computeTensorAddr(
                    parsedCmd.dstBaseAddr, parsedCmd.dstStrideH,
                    parsedCmd.dstStrideW, parsedCmd.dstStrideC, parsedCmd.dstK,
                    parsedCmd.shapeW, parsedCmd.shapeC, parsedCmd.dstCutDim,
                    globalY, globalX, z);
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
DmaUnit::buildTransposeLines()
{
    std::map<Addr, std::vector<SourceCopy>> sourceMap;
    std::map<Addr, std::vector<DestCopy>> destMap;

    const std::array<uint32_t, 3> srcExtents = {
        parsedCmd.shapeH, parsedCmd.shapeW, parsedCmd.shapeC};
    std::array<uint32_t, 3> dstExtents = srcExtents;
    std::swap(dstExtents[parsedCmd.transposeDimA],
              dstExtents[parsedCmd.transposeDimB]);

    const auto denseIndex = [](const std::array<uint32_t, 3> &coords,
                               const std::array<uint32_t, 3> &extents) {
        return (static_cast<size_t>(coords[0]) * extents[1] + coords[1]) *
            extents[2] + coords[2];
    };

    for (uint32_t y = 0; y < parsedCmd.shapeH; ++y) {
        for (uint32_t x = 0; x < parsedCmd.shapeW; ++x) {
            for (uint32_t z = 0; z < parsedCmd.shapeC; ++z) {
                const std::array<uint32_t, 3> srcCoords = {y, x, z};
                auto dstCoords = srcCoords;
                std::swap(dstCoords[parsedCmd.transposeDimA],
                          dstCoords[parsedCmd.transposeDimB]);

                const size_t srcOffset = denseIndex(srcCoords, srcExtents);
                const size_t dstOffset = denseIndex(dstCoords, dstExtents);

                const Addr srcAddr = computeTensorAddr(
                    parsedCmd.srcBaseAddr, parsedCmd.srcStrideH,
                    parsedCmd.srcStrideW, parsedCmd.srcStrideC, 0,
                    parsedCmd.shapeW, parsedCmd.shapeC,
                    static_cast<uint8_t>(CutDim::W), y, x, z);
                const Addr srcLineAddr = srcAddr & ~(CacheLineBytes - 1);
                validateBurstLine(srcLineAddr, sourceSpace(), "source");
                sourceMap[srcLineAddr].push_back({
                    srcOffset,
                    static_cast<uint8_t>(srcAddr - srcLineAddr),
                });

                const Addr dstAddr = computeTensorAddr(
                    parsedCmd.dstBaseAddr, parsedCmd.dstStrideH,
                    parsedCmd.dstStrideW, parsedCmd.dstStrideC, 0,
                    parsedCmd.shapeW, parsedCmd.shapeC,
                    static_cast<uint8_t>(CutDim::W),
                    dstCoords[0], dstCoords[1], dstCoords[2]);
                const Addr dstLineAddr = dstAddr & ~(CacheLineBytes - 1);
                validateBurstLine(dstLineAddr, destSpace(), "destination");
                destMap[dstLineAddr].push_back({
                    static_cast<uint8_t>(dstAddr - dstLineAddr),
                    dstOffset,
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

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Fill) {
        return;
    }

    const PortID readPort = 0;
    uint64_t token = nextMemTxnToken;
    for (size_t i = 0; i < batchPlan.sourceLines.size(); ++i) {
        const auto &line = batchPlan.sourceLines[i];
        MemRequestDesc req;
        req.portId = readPort;
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
        req.portId = readPort;
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
        if (static_cast<Mode>(parsedCmd.mode) == Mode::Transpose) {
            auto &srcBank = bankWorkspace.at(parsedCmd.srcBankId);
            for (const auto &copy : line.copies) {
                srcBank[copy.bufferOffset] = data[copy.lineOffset];
            }
        } else {
            auto &workspace = bankWorkspace.front();
            for (const auto &copy : line.copies) {
                workspace[copy.bufferOffset] = data[copy.lineOffset];
            }
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

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Fill) {
        if (parsedCmd.dstMemSpace == MemorySpace::DmaBank) {
            auto &bank = bankWorkspace.at(parsedCmd.dstBankId);
            bank.assign(bankSize, 0);

            for (uint32_t y = 0; y < parsedCmd.shapeH; ++y) {
                for (uint32_t x = 0; x < parsedCmd.shapeW; ++x) {
                    for (uint32_t z = 0; z < parsedCmd.shapeC; ++z) {
                        const size_t bankOffset = computeTensorAddr(
                            0, parsedCmd.dstStrideH, parsedCmd.dstStrideW,
                            parsedCmd.dstStrideC, 0, parsedCmd.shapeW,
                            parsedCmd.shapeC,
                            static_cast<uint8_t>(CutDim::W), y, x, z);
                        bank[bankOffset] = parsedCmd.fillValue;
                    }
                }
            }

            const size_t requiredBytes = fillRequiredBytes(parsedCmd);
            unsigned long long checksum = 0;
            for (size_t i = 0; i < requiredBytes; ++i) {
                checksum += bank[i];
            }

            DPRINTF(DmaUnit,
                    "DMA_BANK_FILL_OBSERVE bank=%u value=%u required=%llu "
                    "checksum=%llu\n",
                    parsedCmd.dstBankId, parsedCmd.fillValue,
                    static_cast<unsigned long long>(requiredBytes), checksum);
        }
        return 0;
    }

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Transpose) {
        auto &srcBank = bankWorkspace.at(parsedCmd.srcBankId);
        auto &dstBank = bankWorkspace.at(parsedCmd.dstBankId);
        const std::array<uint32_t, 3> srcExtents = {
            parsedCmd.shapeH, parsedCmd.shapeW, parsedCmd.shapeC};
        std::array<uint32_t, 3> dstExtents = srcExtents;
        std::swap(dstExtents[parsedCmd.transposeDimA],
                  dstExtents[parsedCmd.transposeDimB]);

        const auto denseIndex = [](const std::array<uint32_t, 3> &coords,
                                   const std::array<uint32_t, 3> &extents) {
            return (static_cast<size_t>(coords[0]) * extents[1] + coords[1]) *
                extents[2] + coords[2];
        };

        for (uint32_t y = 0; y < parsedCmd.shapeH; ++y) {
            for (uint32_t x = 0; x < parsedCmd.shapeW; ++x) {
                for (uint32_t z = 0; z < parsedCmd.shapeC; ++z) {
                    const std::array<uint32_t, 3> srcCoords = {y, x, z};
                    auto dstCoords = srcCoords;
                    std::swap(dstCoords[parsedCmd.transposeDimA],
                              dstCoords[parsedCmd.transposeDimB]);

                    const size_t srcIndex = denseIndex(srcCoords, srcExtents);
                    const size_t dstIndex = denseIndex(dstCoords, dstExtents);
                    const uint8_t value = srcBank[srcIndex];
                    dstBank[dstIndex] = value;
                    batchPlan.buffer[dstIndex] = value;
                }
            }
        }

        for (auto &line : batchPlan.destLines) {
            for (const auto &copy : line.copies) {
                line.lineData[copy.lineOffset] =
                    batchPlan.buffer[copy.bufferOffset];
            }
        }

        return transposeUnitLatency *
            static_cast<Tick>(axisExtent(parsedCmd, parsedCmd.transposeDimA)) *
            static_cast<Tick>(axisExtent(parsedCmd, parsedCmd.transposeDimB));
    }

    auto &workspace = bankWorkspace.front();
    for (auto &line : batchPlan.destLines) {
        for (const auto &copy : line.copies) {
            line.lineData[copy.lineOffset] = workspace[copy.bufferOffset];
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

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Fill &&
        parsedCmd.dstMemSpace == MemorySpace::DmaBank) {
        return;
    }

    const PortID writePort = memSidePorts.size() > 1 ? 1 : 0;
    for (const auto &line : batchPlan.destLines) {
        MemRequestDesc req;
        req.portId = writePort;
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

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Fill ||
        static_cast<Mode>(parsedCmd.mode) == Mode::Transpose) {
        currentY = 1;
        return;
    }

    if (batchPlan.height != 0 && batchPlan.width != 0) {
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
