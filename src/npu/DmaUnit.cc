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

#include "npu/DmaUnit.hh"

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
      dmaBanks(params.num_banks, std::vector<uint8_t>(params.bank_size, 0))
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

    parsed.stage = static_cast<CommandStage>(opCode & 0x3U);
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
    parsed.srcBankId = parsed.bankCfg & 0xf;
    parsed.dstBankId = (parsed.bankCfg >> 4) & 0xf;

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
            parsed.dstBankId = parsed.bankCfg & 0xfU;
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
    panic_if(cmd.stage > CommandStage::Store,
             "DmaUnit: unsupported command stage=%u",
             static_cast<unsigned>(cmd.stage));

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
             cmd.stage == CommandStage::Legacy ?
                 "DmaUnit: move_layout requires bank_cfg == 0" :
                 "DmaUnit: staged move_layout requires bank_cfg == 0");

    validateBaseAddress(cmd.srcBaseAddr, cmd.srcMemSpace, "source");
    validateBaseAddress(cmd.dstBaseAddr, cmd.dstMemSpace, "destination");

    auto validateBlockedK = [&](uint16_t k, uint8_t cutDim,
                                const char *label) {
        if (k == 0) {
            return;
        }

        switch (static_cast<CutDim>(cutDim)) {
          case CutDim::H:
          {
            const bool invalid = (cmd.shapeH % k) != 0;
            panic_if(invalid,
                     "DmaUnit: %s blocked layout requires H %% k == 0",
                     label);
            return;
          }
          case CutDim::W:
          {
            const bool invalid = (cmd.shapeW % k) != 0;
            panic_if(invalid,
                     "DmaUnit: %s blocked layout requires W %% k == 0",
                     label);
            return;
          }
          case CutDim::C:
          {
            const bool invalid = (cmd.shapeC % k) != 0;
            panic_if(invalid,
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
    panic_if(cmd.stage != CommandStage::Legacy,
             "DmaUnit: staged transpose is unsupported");
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
    panic_if(cmd.stage != CommandStage::Legacy,
             "DmaUnit: staged fill is unsupported");
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
DmaUnit::sourceSpace(const ParsedCmd &cmd) const
{
    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::MoveLayout:
      case Mode::Transpose:
        return cmd.srcMemSpace;
      case Mode::Fill:
        panic("DmaUnit: fill mode has no external source space");
    }

    panic("DmaUnit: unreachable source mode");
}

DmaUnit::MemorySpace
DmaUnit::destSpace(const ParsedCmd &cmd) const
{
    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::MoveLayout:
      case Mode::Transpose:
        return cmd.dstMemSpace;
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

DmaUnit::DmaMacroState &
DmaUnit::macroState(uint64_t macroCmdId)
{
    auto it = macroStates.find(macroCmdId);
    panic_if(it == macroStates.end(),
             "%s: missing DMA macro state for macro %llu", name(),
             static_cast<unsigned long long>(macroCmdId));
    return it->second;
}

const DmaUnit::DmaMacroState &
DmaUnit::macroState(uint64_t macroCmdId) const
{
    auto it = macroStates.find(macroCmdId);
    panic_if(it == macroStates.end(),
             "%s: missing DMA macro state for macro %llu", name(),
             static_cast<unsigned long long>(macroCmdId));
    return it->second;
}

DmaUnit::IterationPlan &
DmaUnit::iterationPlan(DmaMacroState &state, size_t iteration)
{
    panic_if(iteration >= state.iterationPlans.size(),
             "%s: iteration %llu out of range (num plans=%llu)", name(),
             static_cast<unsigned long long>(iteration),
             static_cast<unsigned long long>(state.iterationPlans.size()));
    return state.iterationPlans.at(iteration);
}

const DmaUnit::IterationPlan *
DmaUnit::findIterationPlan(const DmaMacroState &state, size_t iteration) const
{
    if (iteration >= state.iterationPlans.size()) {
        return nullptr;
    }

    return &state.iterationPlans.at(iteration);
}

void
DmaUnit::buildIterationPlans(DmaMacroState &state) const
{
    state.iterationPlans.clear();

    switch (static_cast<Mode>(state.parsedCmd.mode)) {
      case Mode::MoveLayout:
        buildMoveLayoutPlans(state);
        break;
      case Mode::Transpose:
        buildTransposePlans(state);
        break;
      case Mode::Fill:
        buildFillPlans(state);
        break;
    }
}

void
DmaUnit::buildMoveLayoutPlans(DmaMacroState &state) const
{
    const auto &cmd = state.parsedCmd;
    if (cmd.shapeH == 0 || cmd.shapeW == 0 || cmd.shapeC == 0) {
        return;
    }

    const size_t channels = cmd.shapeC;
    panic_if(channels > bankSize,
             "DmaUnit: bank_size=%zu is too small for a (1,1,C) tile",
             bankSize);

    size_t nextBankOffset = 0;
    uint32_t currentY = 0;
    uint32_t currentX = 0;
    while (currentY < cmd.shapeH) {
        IterationPlan plan;
        plan.bankOffset = nextBankOffset;
        plan.startY = currentY;
        plan.startX = currentX;

        const uint32_t remainingH = cmd.shapeH - currentY;
        const uint32_t remainingW = cmd.shapeW - currentX;

        if (currentX == 0) {
            const size_t hSliceBytes =
                static_cast<size_t>(cmd.shapeW) * cmd.shapeC;
            if (hSliceBytes <= bankSize) {
                plan.height = std::max<uint32_t>(
                    1, std::min<uint32_t>(remainingH, bankSize / hSliceBytes));
                plan.width = cmd.shapeW;
            } else {
                plan.height = 1;
                plan.width = std::max<uint32_t>(
                    1, std::min<uint32_t>(remainingW, bankSize / channels));
            }
        } else {
            plan.height = 1;
            plan.width = std::max<uint32_t>(
                1, std::min<uint32_t>(remainingW, bankSize / channels));
        }

        panic_if(plan.width == 0,
                 "DmaUnit: failed to plan a non-empty batch");

        const size_t batchBytes = static_cast<size_t>(plan.height) *
                                  plan.width * cmd.shapeC;
        panic_if(batchBytes > bankSize,
                 "DmaUnit: planned move_layout batch requires %zu bytes, "
                 "exceeds bank_size=%zu",
                 batchBytes, bankSize);

        plan.sourceBuffer.assign(batchBytes, 0);
        buildBatchLines(cmd, plan);
        state.iterationPlans.push_back(std::move(plan));
        nextBankOffset += batchBytes;

        const auto &finished = state.iterationPlans.back();
        DPRINTF(DmaUnit,
                "Planned batch iter=%llu y=%u x=%u h=%u w=%u "
                "src_lines=%u dst_lines=%u\n",
                static_cast<unsigned long long>(state.iterationPlans.size() - 1),
                finished.startY, finished.startX, finished.height,
                finished.width,
                static_cast<unsigned>(finished.sourceLines.size()),
                static_cast<unsigned>(finished.destLines.size()));

        if (finished.width == cmd.shapeW && finished.startX == 0) {
            currentY += finished.height;
            currentX = 0;
        } else {
            currentX += finished.width;
            if (currentX >= cmd.shapeW) {
                currentX = 0;
                currentY += 1;
            }
        }
    }
}

void
DmaUnit::buildTransposePlans(DmaMacroState &state) const
{
    const auto &cmd = state.parsedCmd;
    if (cmd.shapeH == 0 || cmd.shapeW == 0 || cmd.shapeC == 0) {
        return;
    }

    const uint8_t remainingDim =
        3 - cmd.transposeDimA - cmd.transposeDimB;
    const Tick extentA =
        static_cast<Tick>(axisExtent(cmd, cmd.transposeDimA));
    const Tick extentB =
        static_cast<Tick>(axisExtent(cmd, cmd.transposeDimB));
    const Tick extentRest =
        static_cast<Tick>(axisExtent(cmd, remainingDim));
    const Tick totalLatency =
        transposeUnitLatency * extentA * extentB * extentRest;

    DPRINTF(DmaUnit,
            "DMA_TRANSPOSE_LATENCY dim_a=%u dim_b=%u extent_a=%llu "
            "extent_b=%llu extent_rest=%llu transpose_unit_latency=%llu "
            "computed_total_latency=%llu\n",
            static_cast<unsigned>(cmd.transposeDimA),
            static_cast<unsigned>(cmd.transposeDimB),
            static_cast<unsigned long long>(extentA),
            static_cast<unsigned long long>(extentB),
            static_cast<unsigned long long>(extentRest),
            static_cast<unsigned long long>(transposeUnitLatency),
            static_cast<unsigned long long>(totalLatency));

    const size_t planeBytes =
        static_cast<size_t>(extentA) * static_cast<size_t>(extentB);
    for (uint32_t rest = 0; rest < extentRest; ++rest) {
        IterationPlan plan;
        plan.bankOffset = static_cast<size_t>(rest) * planeBytes;
        plan.transposeRemainingDim = remainingDim;
        plan.transposeRemainingIndex = rest;
        plan.execLatency = transposeUnitLatency * extentA * extentB;
        plan.sourceBuffer.assign(planeBytes, 0);
        plan.buffer.assign(planeBytes, 0);
        buildTransposeLines(cmd, plan);
        state.iterationPlans.push_back(std::move(plan));
    }
}

void
DmaUnit::buildFillPlans(DmaMacroState &state) const
{
    const auto &cmd = state.parsedCmd;
    IterationPlan plan;
    if (cmd.dstMemSpace == MemorySpace::DmaBank) {
        state.iterationPlans.push_back(std::move(plan));
        return;
    }

    for (Addr lineAddr : externalFillLineAddrs(cmd)) {
        DestLine line;
        line.lineAddr = lineAddr;
        line.lineData.fill(cmd.fillValue);
        plan.destLines.push_back(line);
    }

    if (!plan.destLines.empty()) {
        state.iterationPlans.push_back(std::move(plan));
    }
}

void
DmaUnit::buildBatchLines(const ParsedCmd &cmd, IterationPlan &plan) const
{
    std::map<Addr, std::vector<SourceCopy>> sourceMap;
    std::map<Addr, std::vector<DestCopy>> destMap;

    for (uint32_t localY = 0; localY < plan.height; ++localY) {
        for (uint32_t localX = 0; localX < plan.width; ++localX) {
            for (uint32_t z = 0; z < cmd.shapeC; ++z) {
                const uint32_t globalY = plan.startY + localY;
                const uint32_t globalX = plan.startX + localX;
                const size_t bufferOffset =
                    (static_cast<size_t>(localY) * plan.width + localX) *
                        cmd.shapeC +
                    z;

                const Addr srcAddr = computeTensorAddr(
                    cmd.srcBaseAddr, cmd.srcStrideH,
                    cmd.srcStrideW, cmd.srcStrideC, cmd.srcK,
                    cmd.shapeW, cmd.shapeC, cmd.srcCutDim,
                    globalY, globalX, z);
                const Addr srcLineAddr = srcAddr & ~(CacheLineBytes - 1);
                validateBurstLine(srcLineAddr, sourceSpace(cmd), "source");
                sourceMap[srcLineAddr].push_back({
                    bufferOffset,
                    static_cast<uint8_t>(srcAddr - srcLineAddr),
                });

                const Addr dstAddr = computeTensorAddr(
                    cmd.dstBaseAddr, cmd.dstStrideH,
                    cmd.dstStrideW, cmd.dstStrideC, cmd.dstK,
                    cmd.shapeW, cmd.shapeC, cmd.dstCutDim,
                    globalY, globalX, z);
                const Addr dstLineAddr = dstAddr & ~(CacheLineBytes - 1);
                validateBurstLine(dstLineAddr, destSpace(cmd), "destination");
                destMap[dstLineAddr].push_back({
                    static_cast<uint8_t>(dstAddr - dstLineAddr),
                    bufferOffset,
                });
            }
        }
    }

    for (const auto &entry : sourceMap) {
        plan.sourceLines.push_back({entry.first, entry.second});
    }
    for (const auto &entry : destMap) {
        plan.destLines.push_back({entry.first, entry.second, {}});
    }
}

void
DmaUnit::buildTransposeLines(const ParsedCmd &cmd, IterationPlan &plan) const
{
    std::map<Addr, std::vector<SourceCopy>> sourceMap;
    std::map<Addr, std::vector<DestCopy>> destMap;

    const uint32_t extentA = axisExtent(cmd, cmd.transposeDimA);
    const uint32_t extentB = axisExtent(cmd, cmd.transposeDimB);
    const uint8_t remainingDim = plan.transposeRemainingDim;

    for (uint32_t a = 0; a < extentA; ++a) {
        for (uint32_t b = 0; b < extentB; ++b) {
            std::array<uint32_t, 3> srcCoords = {0, 0, 0};
            srcCoords[cmd.transposeDimA] = a;
            srcCoords[cmd.transposeDimB] = b;
            srcCoords[remainingDim] = plan.transposeRemainingIndex;

            auto dstCoords = srcCoords;
            std::swap(dstCoords[cmd.transposeDimA],
                      dstCoords[cmd.transposeDimB]);

            const size_t srcOffset =
                static_cast<size_t>(a) * extentB + b;
            const size_t dstOffset =
                static_cast<size_t>(dstCoords[cmd.transposeDimA]) * extentA +
                dstCoords[cmd.transposeDimB];

            const Addr srcAddr = computeTensorAddr(
                cmd.srcBaseAddr, cmd.srcStrideH,
                cmd.srcStrideW, cmd.srcStrideC, 0,
                cmd.shapeW, cmd.shapeC,
                static_cast<uint8_t>(CutDim::W),
                srcCoords[0], srcCoords[1], srcCoords[2]);
            const Addr srcLineAddr = srcAddr & ~(CacheLineBytes - 1);
            validateBurstLine(srcLineAddr, sourceSpace(cmd), "source");
            sourceMap[srcLineAddr].push_back({
                srcOffset,
                static_cast<uint8_t>(srcAddr - srcLineAddr),
            });

            const Addr dstAddr = computeTensorAddr(
                cmd.dstBaseAddr, cmd.dstStrideH,
                cmd.dstStrideW, cmd.dstStrideC, 0,
                cmd.shapeW, cmd.shapeC,
                static_cast<uint8_t>(CutDim::W),
                dstCoords[0], dstCoords[1], dstCoords[2]);
            const Addr dstLineAddr = dstAddr & ~(CacheLineBytes - 1);
            validateBurstLine(dstLineAddr, destSpace(cmd), "destination");
            destMap[dstLineAddr].push_back({
                static_cast<uint8_t>(dstAddr - dstLineAddr),
                dstOffset,
            });
        }
    }

    for (const auto &entry : sourceMap) {
        plan.sourceLines.push_back({entry.first, entry.second});
    }
    for (const auto &entry : destMap) {
        plan.destLines.push_back({entry.first, entry.second, {}});
    }
}

bool
DmaUnit::destLineHazard(const IterationPlan &loadPlan,
                        const IterationPlan &storePlan) const
{
    for (const auto &load_line : loadPlan.destLines) {
        for (const auto &store_line : storePlan.destLines) {
            if (load_line.lineAddr == store_line.lineAddr) {
                return true;
            }
        }
    }

    return false;
}

PortID
DmaUnit::readPortId() const
{
    return 0;
}

PortID
DmaUnit::writePortId() const
{
    return memSidePorts.size() > 1 ? 1 : 0;
}

bool
DmaUnit::isStagedMoveLayout(const ParsedCmd &cmd) const
{
    return static_cast<Mode>(cmd.mode) == Mode::MoveLayout &&
           cmd.stage != CommandStage::Legacy;
}

bool
DmaUnit::stagedMoveLayoutDescriptorsMatch(const ParsedCmd &lhs,
                                          const ParsedCmd &rhs) const
{
    return lhs.deviceId == rhs.deviceId &&
           lhs.dataType == rhs.dataType &&
           lhs.mode == rhs.mode &&
           lhs.srcMemSpace == rhs.srcMemSpace &&
           lhs.dstMemSpace == rhs.dstMemSpace &&
           lhs.srcCutDim == rhs.srcCutDim &&
           lhs.dstCutDim == rhs.dstCutDim &&
           lhs.modeCfg == rhs.modeCfg &&
           lhs.bankCfg == rhs.bankCfg &&
           lhs.word15 == rhs.word15 &&
           lhs.fillValue == rhs.fillValue &&
           lhs.srcBaseAddr == rhs.srcBaseAddr &&
           lhs.dstBaseAddr == rhs.dstBaseAddr &&
           lhs.shapeH == rhs.shapeH &&
           lhs.shapeW == rhs.shapeW &&
           lhs.shapeC == rhs.shapeC &&
           lhs.srcStrideH == rhs.srcStrideH &&
           lhs.srcStrideW == rhs.srcStrideW &&
           lhs.srcStrideC == rhs.srcStrideC &&
           lhs.dstStrideH == rhs.dstStrideH &&
           lhs.dstStrideW == rhs.dstStrideW &&
           lhs.dstStrideC == rhs.dstStrideC &&
           lhs.srcK == rhs.srcK &&
           lhs.dstK == rhs.dstK;
}

DmaUnit::StagedMoveLayoutSequence &
DmaUnit::stagedMoveLayoutSequence()
{
    panic_if(!stagedMoveLayoutState.has_value(),
             "%s: missing staged move_layout sequence state", name());
    return *stagedMoveLayoutState;
}

const DmaUnit::StagedMoveLayoutSequence &
DmaUnit::stagedMoveLayoutSequence() const
{
    panic_if(!stagedMoveLayoutState.has_value(),
             "%s: missing staged move_layout sequence state", name());
    return *stagedMoveLayoutState;
}

void
DmaUnit::registerStagedMoveLayoutMacro(MacroCmdContext &macroCmd,
                                       DmaMacroState &state)
{
    const auto &cmd = state.parsedCmd;
    panic_if(!isStagedMoveLayout(cmd),
             "%s: staged move_layout registration requires a staged "
             "move_layout command",
             name());

    if (!stagedMoveLayoutState.has_value()) {
        panic_if(cmd.stage != CommandStage::Load,
                 "DmaUnit: staged move_layout sequence must start with Load");

        DmaMacroState plan_state;
        plan_state.parsedCmd = cmd;
        buildMoveLayoutPlans(plan_state);

        StagedMoveLayoutSequence seq;
        seq.descriptor = cmd;
        seq.iterationPlans = std::move(plan_state.iterationPlans);
        seq.loadMacroId = macroCmd.macroCmdId;
        stagedMoveLayoutState = std::move(seq);
        state.stagedMoveLayout = true;
        return;
    }

    auto &seq = stagedMoveLayoutSequence();
    panic_if(!stagedMoveLayoutDescriptorsMatch(seq.descriptor, cmd),
             "DmaUnit: staged move_layout command does not match the "
             "active staged sequence");

    switch (cmd.stage) {
      case CommandStage::Load:
        panic_if(seq.loadMacroId.has_value(),
                 "DmaUnit: staged move_layout already has a Load command");
        seq.loadMacroId = macroCmd.macroCmdId;
        break;
      case CommandStage::Compute:
        panic_if(seq.computeMacroId.has_value(),
                 "DmaUnit: staged move_layout already has a Compute command");
        seq.computeMacroId = macroCmd.macroCmdId;
        break;
      case CommandStage::Store:
        panic_if(seq.storeMacroId.has_value(),
                 "DmaUnit: staged move_layout already has a Store command");
        seq.storeMacroId = macroCmd.macroCmdId;
        break;
      case CommandStage::Legacy:
        panic("%s: staged move_layout registration reached Legacy stage",
              name());
    }

    state.stagedMoveLayout = true;
}

bool
DmaUnit::needsSourceLoads(const ParsedCmd &cmd) const
{
    return static_cast<Mode>(cmd.mode) != Mode::Fill;
}

bool
DmaUnit::needsDestLoads(const ParsedCmd &cmd) const
{
    return static_cast<Mode>(cmd.mode) != Mode::Fill;
}

bool
DmaUnit::needsStores(const ParsedCmd &cmd) const
{
    return !(static_cast<Mode>(cmd.mode) == Mode::Fill &&
             cmd.dstMemSpace == MemorySpace::DmaBank);
}

DmaUnit::DmaMacroState::RuntimeStage
DmaUnit::initialRuntimeStage(const ParsedCmd &cmd) const
{
    if (needsSourceLoads(cmd)) {
        return DmaMacroState::RuntimeStage::SourceLoads;
    }
    if (needsDestLoads(cmd)) {
        return DmaMacroState::RuntimeStage::DestLoads;
    }
    return DmaMacroState::RuntimeStage::Compute;
}

const char *
DmaUnit::stageName(CommandStage stage) const
{
    switch (stage) {
      case CommandStage::Legacy:
        return "Legacy";
      case CommandStage::Load:
        return "Load";
      case CommandStage::Compute:
        return "Compute";
      case CommandStage::Store:
        return "Store";
    }

    panic("%s: unreachable DMA stage name", name());
}

const char *
DmaUnit::modeName(Mode mode) const
{
    switch (mode) {
      case Mode::MoveLayout:
        return "MoveLayout";
      case Mode::Transpose:
        return "Transpose";
      case Mode::Fill:
        return "Fill";
    }

    panic("%s: unreachable DMA mode name", name());
}

const char *
DmaUnit::memorySpaceName(MemorySpace space) const
{
    switch (space) {
      case MemorySpace::Dram:
        return "Dram";
      case MemorySpace::Spm:
        return "Spm";
      case MemorySpace::DmaBank:
        return "DmaBank";
      case MemorySpace::Invalid:
        return "Invalid";
    }

    panic("%s: unreachable DMA memory-space name", name());
}

const char *
DmaUnit::cutDimName(uint8_t dim) const
{
    switch (static_cast<CutDim>(dim)) {
      case CutDim::H:
        return "H";
      case CutDim::W:
        return "W";
      case CutDim::C:
        return "C";
      case CutDim::Reserved:
        return "Reserved";
    }

    panic("%s: unreachable DMA cut-dim name", name());
}

void
DmaUnit::appendReadUop(MacroCmdContext &macroCmd, DmaMacroState &state,
                       size_t iteration, PendingMvinKind kind, size_t index,
                       Addr addr, size_t size)
{
    MicroOpContext uop;
    uop.kind = MicroOpContext::Kind::Load;
    uop.macroCmdId = macroCmd.macroCmdId;
    uop.ownerIssueQueueId = macroCmd.targetIssueQueueId;
    uop.portId = readPortId();
    uop.token = nextMicroOpToken++;
    uop.addr = addr;
    uop.size = size;
    macroCmd.uopQueue.push_back(std::move(uop));
    auto &queued = macroCmd.uopQueue.back();
    state.pendingMvinTxns.emplace(queued.token, PendingMvinTxn{
        iteration,
        kind,
        index,
    });
}

void
DmaUnit::appendWriteUop(MacroCmdContext &macroCmd, Addr addr, size_t size,
                        const std::vector<uint8_t> &data)
{
    MicroOpContext uop;
    uop.kind = MicroOpContext::Kind::Store;
    uop.macroCmdId = macroCmd.macroCmdId;
    uop.ownerIssueQueueId = macroCmd.targetIssueQueueId;
    uop.portId = writePortId();
    uop.token = nextMicroOpToken++;
    uop.addr = addr;
    uop.size = size;
    uop.data = data;
    macroCmd.uopQueue.push_back(std::move(uop));
}

void
DmaUnit::queueStagedMoveLayoutWork(MacroCmdContext &macroCmd,
                                   DmaMacroState &state)
{
    auto &seq = stagedMoveLayoutSequence();
    while (macroCmd.uopQueue.empty()) {
        switch (state.parsedCmd.stage) {
          case CommandStage::Load:
          {
            if (seq.nextLoadIteration >= seq.iterationPlans.size()) {
                markEpiloguePending(macroCmd);
                return;
            }
            if (seq.loadedIteration.has_value()) {
                return;
            }

            const size_t iteration = seq.nextLoadIteration;
            const auto &plan = seq.iterationPlans.at(iteration);
            if (seq.computedIteration.has_value() &&
                destLineHazard(plan,
                               seq.iterationPlans.at(*seq.computedIteration))) {
                return;
            }
            state.stagedIterationInFlight = iteration;
            for (size_t i = 0; i < plan.sourceLines.size(); ++i) {
                appendReadUop(macroCmd, state, iteration,
                              PendingMvinKind::SourceLine, i,
                              plan.sourceLines[i].lineAddr, CacheLineBytes);
            }
            for (size_t i = 0; i < plan.destLines.size(); ++i) {
                appendReadUop(macroCmd, state, iteration,
                              PendingMvinKind::DestLine, i,
                              plan.destLines[i].lineAddr, CacheLineBytes);
            }
            if (!macroCmd.uopQueue.empty()) {
                return;
            }

            seq.loadedIteration = iteration;
            seq.nextLoadIteration++;
            state.stagedIterationInFlight.reset();
            continue;
          }

          case CommandStage::Compute:
          {
            if (seq.nextComputeIteration >= seq.iterationPlans.size()) {
                markEpiloguePending(macroCmd);
                return;
            }
            if (!seq.loadedIteration.has_value() ||
                *seq.loadedIteration != seq.nextComputeIteration ||
                seq.computedIteration.has_value()) {
                return;
            }

            const size_t iteration = seq.nextComputeIteration;
            state.stagedIterationInFlight = iteration;
            appendExecUop(macroCmd,
                          std::max<Tick>(1,
                              seq.iterationPlans.at(iteration).execLatency));
            macroCmd.uopQueue.back().token = iteration;
            return;
          }

          case CommandStage::Store:
          {
            if (seq.nextStoreIteration >= seq.iterationPlans.size()) {
                markEpiloguePending(macroCmd);
                return;
            }
            if (!seq.computedIteration.has_value() ||
                *seq.computedIteration != seq.nextStoreIteration) {
                return;
            }

            const size_t iteration = seq.nextStoreIteration;
            const auto &plan = seq.iterationPlans.at(iteration);
            state.stagedIterationInFlight = iteration;
            for (const auto &line : plan.destLines) {
                appendWriteUop(macroCmd, line.lineAddr, CacheLineBytes,
                               std::vector<uint8_t>(line.lineData.begin(),
                                                    line.lineData.end()));
            }
            if (!macroCmd.uopQueue.empty()) {
                return;
            }

            panic_if(!seq.computedIteration.has_value() ||
                         *seq.computedIteration != iteration,
                     "%s: staged move_layout store lost computed batch %llu",
                     name(), static_cast<unsigned long long>(iteration));
            seq.computedIteration.reset();
            seq.nextStoreIteration++;
            if (seq.overlapObservedStoreIteration == iteration) {
                seq.overlapObservedStoreIteration.reset();
            }
            state.stagedIterationInFlight.reset();
            continue;
          }

          case CommandStage::Legacy:
            panic("%s: staged move_layout queue reached Legacy stage",
                  name());
        }
    }
}

void
DmaUnit::wakeStagedMoveLayoutMacros()
{
    if (!stagedMoveLayoutState.has_value()) {
        return;
    }

    auto try_wake = [this](const std::optional<uint64_t> &macro_id_opt) {
        if (!macro_id_opt.has_value()) {
            return;
        }

        auto macro_it = macroCmdContexts.find(*macro_id_opt);
        if (macro_it == macroCmdContexts.end()) {
            return;
        }

        auto &macro_cmd = macro_it->second;
        if (macro_cmd.phase != Phase::Active || macro_cmd.waitingCallback ||
            !macro_cmd.uopQueue.empty()) {
            return;
        }

        auto &state = macroState(*macro_id_opt);
        if (!state.stagedMoveLayout) {
            return;
        }
        queueStagedMoveLayoutWork(macro_cmd, state);
    };

    const auto &seq = stagedMoveLayoutSequence();
    try_wake(seq.computeMacroId);
    try_wake(seq.loadMacroId);
    try_wake(seq.storeMacroId);
    tryScheduleIssue();
}

void
DmaUnit::observeStagedMoveLayoutOverlap(const DmaMacroState &state,
                                        size_t iteration)
{
    if (!stagedMoveLayoutState.has_value()) {
        return;
    }

    auto &seq = stagedMoveLayoutSequence();
    size_t store_iteration = 0;
    std::optional<uint64_t> other_macro_id;
    if (state.parsedCmd.stage == CommandStage::Load) {
        if (iteration == 0) {
            return;
        }
        store_iteration = iteration - 1;
        other_macro_id = seq.storeMacroId;
    } else if (state.parsedCmd.stage == CommandStage::Store) {
        store_iteration = iteration;
        other_macro_id = seq.loadMacroId;
    } else {
        return;
    }

    if (seq.overlapObservedStoreIteration == store_iteration ||
        !other_macro_id.has_value()) {
        return;
    }

    for (const auto &entry : activeMemTxns) {
        const auto &txn = entry.second;
        if (txn.macroCmdId != *other_macro_id) {
            continue;
        }

        const auto &other_state = macroState(txn.macroCmdId);
        if (!other_state.stagedIterationInFlight.has_value()) {
            continue;
        }

        if (state.parsedCmd.stage == CommandStage::Load) {
            if (*other_state.stagedIterationInFlight != store_iteration) {
                continue;
            }
        } else {
            if (*other_state.stagedIterationInFlight != store_iteration + 1) {
                continue;
            }
        }

        seq.overlapObservedStoreIteration = store_iteration;
        observedBatchOverlapCountValue++;
        DPRINTF(DmaUnit,
                "DMA_OVERLAP_OBSERVE store_iter=%llu load_iter=%llu "
                "observed=%llu\n",
                static_cast<unsigned long long>(store_iteration),
                static_cast<unsigned long long>(store_iteration + 1),
                static_cast<unsigned long long>(
                    observedBatchOverlapCountValue));
        return;
    }
}

void
DmaUnit::finishStoreStage(MacroCmdContext &macroCmd, DmaMacroState &state)
{
    (void)macroCmd;
    state.currentIteration++;
    state.runtimeStage = initialRuntimeStage(state.parsedCmd);
}

void
DmaUnit::queueNextStage(MacroCmdContext &macroCmd, DmaMacroState &state)
{
    while (macroCmd.uopQueue.empty()) {
        const IterationPlan *plan_ptr =
            findIterationPlan(state, state.currentIteration);
        if (plan_ptr == nullptr) {
            markEpiloguePending(macroCmd);
            return;
        }
        const auto &cmd = state.parsedCmd;
        const auto &plan = *plan_ptr;

        switch (state.runtimeStage) {
          case DmaMacroState::RuntimeStage::SourceLoads:
            if (!needsSourceLoads(cmd) || plan.sourceLines.empty()) {
                state.runtimeStage = needsDestLoads(cmd) ?
                    DmaMacroState::RuntimeStage::DestLoads :
                    DmaMacroState::RuntimeStage::Compute;
                continue;
            }
            for (size_t i = 0; i < plan.sourceLines.size(); ++i) {
                appendReadUop(macroCmd, state, state.currentIteration,
                              PendingMvinKind::SourceLine, i,
                              plan.sourceLines[i].lineAddr, CacheLineBytes);
            }
            return;

          case DmaMacroState::RuntimeStage::DestLoads:
            if (!needsDestLoads(cmd) || plan.destLines.empty()) {
                state.runtimeStage = DmaMacroState::RuntimeStage::Compute;
                continue;
            }
            for (size_t i = 0; i < plan.destLines.size(); ++i) {
                appendReadUop(macroCmd, state, state.currentIteration,
                              PendingMvinKind::DestLine, i,
                              plan.destLines[i].lineAddr, CacheLineBytes);
            }
            return;

          case DmaMacroState::RuntimeStage::Compute:
            appendExecUop(macroCmd, std::max<Tick>(1, plan.execLatency));
            macroCmd.uopQueue.back().token = state.currentIteration;
            return;

          case DmaMacroState::RuntimeStage::Stores:
            if (!needsStores(cmd) || plan.destLines.empty()) {
                finishStoreStage(macroCmd, state);
                continue;
            }
            for (const auto &line : plan.destLines) {
                appendWriteUop(macroCmd, line.lineAddr, CacheLineBytes,
                               std::vector<uint8_t>(line.lineData.begin(),
                                                    line.lineData.end()));
            }
            return;
        }
    }
}

void
DmaUnit::materializeBankFill(const ParsedCmd &cmd)
{
    auto &bank = dmaBanks.at(cmd.dstBankId);
    const size_t requiredBytes = fillRequiredBytes(cmd);
    std::fill(bank.begin(), bank.end(), 0);
    std::fill_n(bank.begin(), requiredBytes, cmd.fillValue);

    const unsigned long long checksum =
        static_cast<unsigned long long>(requiredBytes) * cmd.fillValue;
    DPRINTF(DmaUnit,
            "DMA_BANK_FILL_OBSERVE bank=%u value=%u required=%llu "
            "checksum=%llu\n",
            cmd.dstBankId, cmd.fillValue,
            static_cast<unsigned long long>(requiredBytes), checksum);
}

void
DmaUnit::executeIteration(DmaMacroState &state, IterationPlan &plan)
{
    const auto &cmd = state.parsedCmd;

    if (static_cast<Mode>(cmd.mode) == Mode::Fill) {
        if (cmd.dstMemSpace == MemorySpace::DmaBank) {
            materializeBankFill(cmd);
        } else {
            for (auto &line : plan.destLines) {
                line.lineData.fill(cmd.fillValue);
            }
        }
        return;
    }

    if (static_cast<Mode>(cmd.mode) == Mode::Transpose) {
        const uint32_t extentA = axisExtent(cmd, cmd.transposeDimA);
        const uint32_t extentB = axisExtent(cmd, cmd.transposeDimB);
        for (uint32_t a = 0; a < extentA; ++a) {
            for (uint32_t b = 0; b < extentB; ++b) {
                const size_t srcIndex = static_cast<size_t>(a) * extentB + b;
                const size_t dstIndex = static_cast<size_t>(b) * extentA + a;
                plan.buffer[dstIndex] = plan.sourceBuffer[srcIndex];
            }
        }

        for (auto &line : plan.destLines) {
            for (const auto &copy : line.copies) {
                line.lineData[copy.lineOffset] =
                    plan.buffer[copy.bufferOffset];
            }
        }
        return;
    }

    for (auto &line : plan.destLines) {
        for (const auto &copy : line.copies) {
            line.lineData[copy.lineOffset] = plan.sourceBuffer[copy.bufferOffset];
        }
    }
}

SpecializedExecutionUnit::MacroCmdKind
DmaUnit::classifyMacroCmd(const std::vector<uint8_t> &cmd) const
{
    switch (parseCommand(cmd).stage) {
      case CommandStage::Legacy:
      case CommandStage::Compute:
        return MacroCmdKind::Exec;
      case CommandStage::Load:
        return MacroCmdKind::Load;
      case CommandStage::Store:
        return MacroCmdKind::Store;
    }

    panic("%s: unreachable DMA command stage classification", name());
}

uint32_t
DmaUnit::classifyIssueQueue(const std::vector<uint8_t> &cmd,
                            MacroCmdKind kind) const
{
    (void)cmd;
    switch (kind) {
      case MacroCmdKind::Exec:
        return 0;
      case MacroCmdKind::Load:
        return 1 + readPortId();
      case MacroCmdKind::Store:
        return 1 + writePortId();
    }

    panic("%s: unreachable DMA issue-queue classification", name());
}

bool
DmaUnit::canActivateMacroCmd(const MacroCmdContext &macroCmd) const
{
    const ParsedCmd cmd = parseCommand(macroCmd.cmd);
    if (cmd.stage != CommandStage::Legacy &&
        cmd.stage != CommandStage::Compute) {
        return true;
    }

    bool touchesExternalMem = false;
    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::MoveLayout:
      case Mode::Transpose:
        touchesExternalMem =
            isExternalSpace(cmd.srcMemSpace) ||
            isExternalSpace(cmd.dstMemSpace);
        break;
      case Mode::Fill:
        touchesExternalMem = isExternalSpace(cmd.dstMemSpace);
        break;
    }

    if (!touchesExternalMem) {
        return true;
    }

    return canActivateExclusively(macroCmd);
}

void
DmaUnit::onMacroCmdBegin(MacroCmdContext &macroCmd)
{
    DmaMacroState state;
    state.parsedCmd = parseCommand(macroCmd.cmd);
    validateParsedCommand(state.parsedCmd);
    if (stagedMoveLayoutState.has_value() &&
        !isStagedMoveLayout(state.parsedCmd)) {
        panic("DmaUnit: staged move_layout sequence does not allow "
              "interleaved DMA commands");
    }

    macroStates.emplace(macroCmd.macroCmdId, std::move(state));
    auto &stored = macroState(macroCmd.macroCmdId);
    if (isStagedMoveLayout(stored.parsedCmd)) {
        registerStagedMoveLayoutMacro(macroCmd, stored);
        return;
    }

    stored.runtimeStage = initialRuntimeStage(stored.parsedCmd);
    buildIterationPlans(stored);
}

void
DmaUnit::buildUops(MacroCmdContext &macroCmd)
{
    auto &state = macroState(macroCmd.macroCmdId);
    if (state.stagedMoveLayout) {
        queueStagedMoveLayoutWork(macroCmd, state);
        return;
    }
    queueNextStage(macroCmd, state);
}

void
DmaUnit::onMemUopComplete(MacroCmdContext &macroCmd,
                          const MemTxnContext &txn, PacketPtr pkt)
{
    auto &state = macroState(macroCmd.macroCmdId);
    auto pending_it = state.pendingMvinTxns.find(txn.token);
    if (pending_it != state.pendingMvinTxns.end()) {
        const PendingMvinTxn pending = pending_it->second;
        state.pendingMvinTxns.erase(pending_it);
        IterationPlan *plan_ptr = nullptr;
        if (state.stagedMoveLayout) {
            auto &seq = stagedMoveLayoutSequence();
            panic_if(pending.iteration >= seq.iterationPlans.size(),
                     "%s: staged iteration %llu out of range (num plans=%llu)",
                     name(),
                     static_cast<unsigned long long>(pending.iteration),
                     static_cast<unsigned long long>(
                         seq.iterationPlans.size()));
            plan_ptr = &seq.iterationPlans.at(pending.iteration);
        } else {
            plan_ptr = &iterationPlan(state, pending.iteration);
        }
        auto &plan = *plan_ptr;

        switch (pending.kind) {
          case PendingMvinKind::SourceLine: {
            const auto &line = plan.sourceLines.at(pending.index);
            const uint8_t *data = pkt->getConstPtr<uint8_t>();
            for (const auto &copy : line.copies) {
                plan.sourceBuffer[copy.bufferOffset] = data[copy.lineOffset];
            }
            break;
          }
          case PendingMvinKind::DestLine: {
            auto &line = plan.destLines.at(pending.index);
            std::memcpy(line.lineData.data(), pkt->getConstPtr<uint8_t>(),
                        CacheLineBytes);
            break;
          }
        }
    }

    if (state.stagedMoveLayout) {
        panic_if(!state.stagedIterationInFlight.has_value(),
                 "%s: staged move_layout mem completion missing "
                 "iteration tracking",
                 name());
        const size_t iteration = *state.stagedIterationInFlight;
        observeStagedMoveLayoutOverlap(state, iteration);
        if (!macroCmd.uopQueue.empty()) {
            return;
        }

        auto &seq = stagedMoveLayoutSequence();
        switch (state.parsedCmd.stage) {
          case CommandStage::Load:
            panic_if(seq.loadedIteration.has_value(),
                     "%s: staged move_layout read slot is already occupied",
                     name());
            seq.loadedIteration = iteration;
            seq.nextLoadIteration = iteration + 1;
            state.stagedIterationInFlight.reset();
            queueStagedMoveLayoutWork(macroCmd, state);
            wakeStagedMoveLayoutMacros();
            return;

          case CommandStage::Store:
            panic_if(!seq.computedIteration.has_value() ||
                         *seq.computedIteration != iteration,
                     "%s: staged move_layout store completed unexpected "
                     "iteration %llu",
                     name(),
                     static_cast<unsigned long long>(iteration));
            seq.computedIteration.reset();
            seq.nextStoreIteration = iteration + 1;
            if (seq.overlapObservedStoreIteration == iteration) {
                seq.overlapObservedStoreIteration.reset();
            }
            state.stagedIterationInFlight.reset();
            queueStagedMoveLayoutWork(macroCmd, state);
            wakeStagedMoveLayoutMacros();
            return;

          case CommandStage::Compute:
          case CommandStage::Legacy:
            panic("%s: unexpected staged move_layout memory completion stage",
                  name());
        }
    }

    if (!macroCmd.uopQueue.empty()) {
        return;
    }

    switch (state.runtimeStage) {
      case DmaMacroState::RuntimeStage::SourceLoads:
        state.runtimeStage = needsDestLoads(state.parsedCmd) ?
            DmaMacroState::RuntimeStage::DestLoads :
            DmaMacroState::RuntimeStage::Compute;
        break;
      case DmaMacroState::RuntimeStage::DestLoads:
        state.runtimeStage = DmaMacroState::RuntimeStage::Compute;
        break;
      case DmaMacroState::RuntimeStage::Stores:
        finishStoreStage(macroCmd, state);
        break;
      case DmaMacroState::RuntimeStage::Compute:
        return;
    }

    queueNextStage(macroCmd, state);
}

void
DmaUnit::onExecUopComplete(MacroCmdContext &macroCmd,
                           const MicroOpContext &uop)
{
    auto &state = macroState(macroCmd.macroCmdId);
    if (state.stagedMoveLayout) {
        auto &seq = stagedMoveLayoutSequence();
        const size_t iteration = uop.token;
        panic_if(!state.stagedIterationInFlight.has_value() ||
                     *state.stagedIterationInFlight != iteration,
                 "%s: staged move_layout compute completed unexpected "
                 "iteration %llu",
                 name(), static_cast<unsigned long long>(iteration));
        panic_if(!seq.loadedIteration.has_value() ||
                     *seq.loadedIteration != iteration,
                 "%s: staged move_layout compute requires loaded "
                 "iteration %llu",
                 name(), static_cast<unsigned long long>(iteration));
        panic_if(seq.computedIteration.has_value(),
                 "%s: staged move_layout write slot is already occupied",
                 name());

        DmaMacroState exec_state;
        exec_state.parsedCmd = state.parsedCmd;
        auto &plan = seq.iterationPlans.at(iteration);
        executeIteration(exec_state, plan);

        seq.loadedIteration.reset();
        seq.computedIteration = iteration;
        seq.nextComputeIteration = iteration + 1;
        state.stagedIterationInFlight.reset();
        queueStagedMoveLayoutWork(macroCmd, state);
        wakeStagedMoveLayoutMacros();
        return;
    }

    auto &plan = iterationPlan(state, state.currentIteration);
    executeIteration(state, plan);

    state.runtimeStage = DmaMacroState::RuntimeStage::Stores;
    if (!needsStores(state.parsedCmd)) {
        finishStoreStage(macroCmd, state);
    }
    queueNextStage(macroCmd, state);
}

void
DmaUnit::onMacroCmdEnd(MacroCmdContext &macroCmd)
{
    auto state_it = macroStates.find(macroCmd.macroCmdId);
    if (state_it != macroStates.end() && state_it->second.stagedMoveLayout &&
        stagedMoveLayoutState.has_value()) {
        auto &seq = stagedMoveLayoutSequence();
        if (seq.loadMacroId == macroCmd.macroCmdId) {
            seq.loadMacroId.reset();
        }
        if (seq.computeMacroId == macroCmd.macroCmdId) {
            seq.computeMacroId.reset();
        }
        if (seq.storeMacroId == macroCmd.macroCmdId) {
            seq.storeMacroId.reset();
        }
        if (!seq.loadMacroId.has_value() &&
            !seq.computeMacroId.has_value() &&
            !seq.storeMacroId.has_value() &&
            seq.nextStoreIteration >= seq.iterationPlans.size()) {
            stagedMoveLayoutState.reset();
        }
    }
    macroStates.erase(macroCmd.macroCmdId);
}

const char *
DmaUnit::profileSeuType() const
{
    return "DMA";
}

void
DmaUnit::appendProfileDetailsJson(const MacroCmdContext &macroCmd,
                                  std::ostream &os) const
{
    const auto &state = macroState(macroCmd.macroCmdId);
    const auto &cmd = state.parsedCmd;
    size_t iteration_plans = state.iterationPlans.size();
    size_t current_iteration = state.currentIteration;
    if (state.stagedMoveLayout && stagedMoveLayoutState.has_value()) {
        const auto &seq = stagedMoveLayoutSequence();
        iteration_plans = seq.iterationPlans.size();
        switch (cmd.stage) {
          case CommandStage::Load:
            current_iteration = seq.nextLoadIteration;
            break;
          case CommandStage::Compute:
            current_iteration = seq.nextComputeIteration;
            break;
          case CommandStage::Store:
            current_iteration = seq.nextStoreIteration;
            break;
          case CommandStage::Legacy:
            break;
        }
    }

    os << "\"stage\":";
    appendJsonString(os, stageName(cmd.stage));
    os << ",\"mode\":";
    appendJsonString(os, modeName(static_cast<Mode>(cmd.mode)));
    os << ",\"src_mem_space\":";
    appendJsonString(os, memorySpaceName(cmd.srcMemSpace));
    os << ",\"dst_mem_space\":";
    appendJsonString(os, memorySpaceName(cmd.dstMemSpace));
    os << ",\"src_cut_dim\":";
    appendJsonString(os, cutDimName(cmd.srcCutDim));
    os << ",\"dst_cut_dim\":";
    appendJsonString(os, cutDimName(cmd.dstCutDim));
    os << ",\"transpose_dim_a\":";
    appendJsonString(os, cutDimName(cmd.transposeDimA));
    os << ",\"transpose_dim_b\":";
    appendJsonString(os, cutDimName(cmd.transposeDimB));
    os << ",\"src_base_addr\":" << cmd.srcBaseAddr;
    os << ",\"dst_base_addr\":" << cmd.dstBaseAddr;
    os << ",\"shape_h\":" << cmd.shapeH;
    os << ",\"shape_w\":" << cmd.shapeW;
    os << ",\"shape_c\":" << cmd.shapeC;
    os << ",\"src_stride_h\":" << cmd.srcStrideH;
    os << ",\"src_stride_w\":" << cmd.srcStrideW;
    os << ",\"src_stride_c\":" << cmd.srcStrideC;
    os << ",\"dst_stride_h\":" << cmd.dstStrideH;
    os << ",\"dst_stride_w\":" << cmd.dstStrideW;
    os << ",\"dst_stride_c\":" << cmd.dstStrideC;
    os << ",\"src_k\":" << cmd.srcK;
    os << ",\"dst_k\":" << cmd.dstK;
    os << ",\"src_bank_id\":" << static_cast<unsigned>(cmd.srcBankId);
    os << ",\"dst_bank_id\":" << static_cast<unsigned>(cmd.dstBankId);
    os << ",\"mode_cfg\":" << cmd.modeCfg;
    os << ",\"bank_cfg\":" << cmd.bankCfg;
    os << ",\"word15\":" << cmd.word15;
    os << ",\"fill_value\":" << static_cast<unsigned>(cmd.fillValue);
    os << ",\"iteration_plans\":" << iteration_plans;
    os << ",\"current_iteration\":" << current_iteration;
    os << ",\"pending_mvin_txns\":" << state.pendingMvinTxns.size();
}

uint64_t
DmaUnit::observedBatchOverlapCount() const
{
    return observedBatchOverlapCountValue;
}

} // namespace gem5
