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

#include "npu/mega/MpuUnit.hh"

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <utility>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/MpuUnit.hh"

namespace gem5
{

namespace
{

constexpr uint32_t SlotBaseMask = ~0xfU;

} // anonymous namespace

MpuUnit::MpuUnit(const MpuUnitParams &params)
    : SpecializedExecutionUnit(params),
      arrayRows(params.array_rows),
      arrayCols(params.array_cols),
      arrayKDepth(params.array_k_depth),
      loadBaseLatency(params.load_base_latency),
      storeBaseLatency(params.store_base_latency),
      arrayFillLatency(params.array_fill_latency),
      arraySteadyPerK(params.array_steady_per_k),
      arrayDrainLatency(params.array_drain_latency),
      loadBandwidthBytesPerCycle(params.load_bandwidth_bytes_per_cycle),
      storeBandwidthBytesPerCycle(params.store_bandwidth_bytes_per_cycle),
      cReadBaseLatency(params.c_read_base_latency),
      cWriteBaseLatency(params.c_write_base_latency),
      localBankCount(params.local_bank_count),
      localBankGranularityBytes(params.local_bank_granularity_bytes),
      localBankServiceCycles(params.local_bank_service_cycles),
      parsedCmdValid(false),
      loadCount(0),
      computeCount(0),
      storeCount(0),
      tensorLoopCount(0),
      matmulCountValue(0),
      matmulAccCountValue(0),
      tensorLoopExpandedTilesCount(0),
      tensorLoopExpandedAccTilesCount(0),
      totalInternalLoadsValue(0),
      totalInternalComputesValue(0),
      totalInternalStoresValue(0),
      totalTilesValue(0),
      totalAccTilesValue(0),
      stallCyclesWaitingForSPMValue(0),
      stallCyclesWaitingForSlotValue(0),
      computedTotalLatencyValue(0)
{
    fatal_if(macroCmdBytes != 64, "%s: MpuUnit requires 64-byte commands",
             name());
    fatal_if(arrayRows == 0 || arrayCols == 0 || arrayKDepth == 0,
             "%s: array dimensions must be non-zero", name());
    fatal_if(memSidePorts.empty(), "%s: MpuUnit requires at least one mem_side "
             "port", name());
    fatal_if(loadBandwidthBytesPerCycle == 0 || storeBandwidthBytesPerCycle == 0,
             "%s: MPU bandwidth bytes/cycle must be non-zero", name());
    fatal_if(localBankCount == 0 || localBankGranularityBytes == 0 ||
             localBankServiceCycles == 0,
             "%s: MPU local bank model parameters must be non-zero", name());

    slots[SlotA0].kind = SlotKind::A;
    slots[SlotA0].localAddr = 0x100;
    slots[SlotA0].bytes.resize(slotCapacityBytes(SlotA0), 0);
    slots[SlotA1].kind = SlotKind::A;
    slots[SlotA1].localAddr = 0x110;
    slots[SlotA1].bytes.resize(slotCapacityBytes(SlotA1), 0);
    slots[SlotB0].kind = SlotKind::B;
    slots[SlotB0].localAddr = 0x120;
    slots[SlotB0].bytes.resize(slotCapacityBytes(SlotB0), 0);
    slots[SlotB1].kind = SlotKind::B;
    slots[SlotB1].localAddr = 0x130;
    slots[SlotB1].bytes.resize(slotCapacityBytes(SlotB1), 0);
    slots[SlotC0].kind = SlotKind::C;
    slots[SlotC0].localAddr = 0x140;
    slots[SlotC0].bytes.resize(slotCapacityBytes(SlotC0), 0);
    slots[SlotC1].kind = SlotKind::C;
    slots[SlotC1].localAddr = 0x150;
    slots[SlotC1].bytes.resize(slotCapacityBytes(SlotC1), 0);
}

uint32_t
MpuUnit::extractWord(const std::vector<uint8_t> &cmd, size_t index) const
{
    panic_if((index + 1) * sizeof(uint32_t) > cmd.size(),
             "MpuUnit: command word %zu is out of range", index);

    uint32_t word = 0;
    std::memcpy(&word, cmd.data() + (index * sizeof(uint32_t)), sizeof(word));
    return word;
}

MpuUnit::ParsedCmd
MpuUnit::parseCommand(const std::vector<uint8_t> &cmd) const
{
    panic_if(cmd.size() != macroCmdBytes,
             "MpuUnit: expected %u-byte command, got %zu bytes",
             macroCmdBytes, cmd.size());

    ParsedCmd parsed;
    for (size_t i = 0; i < parsed.words.size(); ++i) {
        parsed.words[i] = extractWord(cmd, i);
    }

    const uint32_t header = parsed.words[0];
    const uint8_t op_code = (header >> 16) & 0xff;
    parsed.deviceId = (header >> 24) & 0xf;
    parsed.dataType = (op_code >> 5) & 0x7;
    parsed.mode = (op_code >> 2) & 0x7;
    parsed.syncIndicator = (header >> 8) & 0xff;
    parsed.setIndicatorSns = ((header >> 7) & 0x1) != 0;
    parsed.setIndicatorSnd = ((header >> 6) & 0x1) != 0;

    panic_if(((header >> 28) & 0xf) != MpuDeviceType,
             "MpuUnit: unexpected device_type=%u", (header >> 28) & 0xf);

    switch (static_cast<Mode>(parsed.mode)) {
      case Mode::Load:
        parsed.localAddrA = parsed.words[1];
        parsed.spmAddrA = parsed.words[2];
        parsed.validM = parsed.words[3];
        parsed.validN = parsed.words[4];
        parsed.validK = parsed.words[5];
        parsed.layoutMode = parsed.words[6];
        break;
      case Mode::Compute:
        parsed.localAddrA = parsed.words[1];
        parsed.localAddrB = parsed.words[2];
        parsed.localAddrC = parsed.words[3];
        parsed.validM = parsed.words[4];
        parsed.validN = parsed.words[5];
        parsed.validK = parsed.words[6];
        parsed.subop = static_cast<ComputeSubop>(parsed.words[7]);
        parsed.computeModeFlags = parsed.words[8];
        break;
      case Mode::Store:
        parsed.localAddrC = parsed.words[1];
        parsed.spmAddrC = parsed.words[2];
        parsed.validM = parsed.words[3];
        parsed.validN = parsed.words[4];
        parsed.validK = parsed.words[5];
        parsed.layoutMode = parsed.words[6];
        break;
      case Mode::TensorLoop:
        parsed.localAddrA = parsed.words[1];
        parsed.localAddrB = parsed.words[2];
        parsed.localAddrC = parsed.words[3];
        parsed.spmAddrA = parsed.words[4];
        parsed.spmAddrB = parsed.words[5];
        parsed.spmAddrC = parsed.words[6];
        parsed.validM = parsed.words[7];
        parsed.validN = parsed.words[8];
        parsed.validK = parsed.words[9];
        parsed.subop = static_cast<ComputeSubop>(parsed.words[10]);
        parsed.outerAxis = static_cast<Axis>(parsed.words[11]);
        parsed.outerCount = parsed.words[12];
        parsed.innerAxis = static_cast<Axis>(parsed.words[13]);
        parsed.innerCount = parsed.words[14];
        parsed.outerStepTiles = parsed.words[15] & 0xff;
        parsed.innerStepTiles = (parsed.words[15] >> 8) & 0xff;
        parsed.pingpongA = ((parsed.words[15] >> 16) & 0x1) != 0;
        parsed.pingpongB = ((parsed.words[15] >> 17) & 0x1) != 0;
        parsed.pingpongC = ((parsed.words[15] >> 18) & 0x1) != 0;
        parsed.autoLoadCForAcc = ((parsed.words[15] >> 19) & 0x1) != 0;
        parsed.tensorLayoutA =
            ((parsed.words[15] >> TensorStepCfgLayoutABit) & 0x1) != 0 ?
            static_cast<uint32_t>(LayoutMode::Skewed) :
            static_cast<uint32_t>(LayoutMode::Normal);
        parsed.tensorLayoutB =
            ((parsed.words[15] >> TensorStepCfgLayoutBBit) & 0x1) != 0 ?
            static_cast<uint32_t>(LayoutMode::Skewed) :
            static_cast<uint32_t>(LayoutMode::Normal);
        parsed.tensorLayoutC =
            ((parsed.words[15] >> TensorStepCfgLayoutCBit) & 0x1) != 0 ?
            static_cast<uint32_t>(LayoutMode::Skewed) :
            static_cast<uint32_t>(LayoutMode::Normal);
        break;
      default:
        break;
    }

    return parsed;
}

void
MpuUnit::resetCommandState()
{
    parsedCmd = ParsedCmd();
    parsedCmdValid = false;
    iterationPlans.clear();
    pendingLoadTxns.clear();
    tensorLoopAccumulators.clear();
}

void
MpuUnit::validateValidShape(const ParsedCmd &cmd) const
{
    panic_if(cmd.validM == 0 || cmd.validN == 0 || cmd.validK == 0,
             "MpuUnit: valid_m/n/k must all be greater than zero");
    panic_if(cmd.validM > arrayRows,
             "MpuUnit: valid_m=%u exceeds array_rows=%u",
             cmd.validM, arrayRows);
    panic_if(cmd.validN > arrayCols,
             "MpuUnit: valid_n=%u exceeds array_cols=%u",
             cmd.validN, arrayCols);
    panic_if(cmd.validK > arrayKDepth,
             "MpuUnit: valid_k=%u exceeds array_k_depth=%u",
             cmd.validK, arrayKDepth);
}

bool
MpuUnit::spmContains(Addr addr, size_t size) const
{
    if (size == 0 || addr < SpmBase || addr > SpmEnd) {
        return false;
    }

    const Addr size_minus_one = size - 1;
    if (addr > SpmEnd - size_minus_one) {
        return false;
    }

    return true;
}

void
MpuUnit::validateSpmAddress(Addr addr, size_t size, const char *label) const
{
    panic_if(!spmContains(addr, size), "MpuUnit: invalid %s SPM address %#llx "
             "for size=%llu", label, static_cast<unsigned long long>(addr),
             static_cast<unsigned long long>(size));
}

int
MpuUnit::slotIndexForAddr(uint32_t local_addr) const
{
    const uint32_t base_addr = local_addr & SlotBaseMask;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].localAddr == base_addr) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

int
MpuUnit::alternateSlotIndex(int slot_index) const
{
    switch (slot_index) {
      case SlotA0:
        return SlotA1;
      case SlotA1:
        return SlotA0;
      case SlotB0:
        return SlotB1;
      case SlotB1:
        return SlotB0;
      case SlotC0:
        return SlotC1;
      case SlotC1:
        return SlotC0;
    }

    panic("MpuUnit: no alternate slot for index=%d", slot_index);
}

size_t
MpuUnit::localOffsetBytes(uint32_t local_addr) const
{
    return local_addr & 0xfU;
}

MpuUnit::SlotState &
MpuUnit::slotByIndex(int slot_index)
{
    panic_if(slot_index < 0 ||
             static_cast<size_t>(slot_index) >= slots.size(),
             "MpuUnit: slot index %d out of range", slot_index);
    return slots.at(slot_index);
}

const MpuUnit::SlotState &
MpuUnit::slotByIndex(int slot_index) const
{
    panic_if(slot_index < 0 ||
             static_cast<size_t>(slot_index) >= slots.size(),
             "MpuUnit: slot index %d out of range", slot_index);
    return slots.at(slot_index);
}

void
MpuUnit::validateSlotAddress(uint32_t local_addr, SlotKind expected_kind,
                             const char *label) const
{
    const int slot_index = slotIndexForAddr(local_addr);
    panic_if(slot_index < 0,
             "MpuUnit: invalid %s local address %#x", label, local_addr);
    panic_if(slotByIndex(slot_index).kind != expected_kind,
             "MpuUnit: %s local address %#x targets wrong slot type",
             label, local_addr);
}

void
MpuUnit::validateSlotShape(const SlotState &slot, const ParsedCmd &cmd,
                           const char *label) const
{
    panic_if(slot.shapeM != cmd.validM || slot.shapeN != cmd.validN ||
             slot.shapeK != cmd.validK,
             "MpuUnit: %s slot shape mismatch slot=(%u,%u,%u) cmd=(%u,%u,%u)",
             label, slot.shapeM, slot.shapeN, slot.shapeK, cmd.validM,
             cmd.validN, cmd.validK);
}

size_t
MpuUnit::aTileBytes(uint32_t valid_m, uint32_t valid_k) const
{
    return static_cast<size_t>(valid_m) * valid_k * Int8Bytes;
}

size_t
MpuUnit::bTileBytes(uint32_t valid_k, uint32_t valid_n) const
{
    return static_cast<size_t>(valid_k) * valid_n * Int8Bytes;
}

size_t
MpuUnit::cTileBytes(uint32_t valid_m, uint32_t valid_n) const
{
    return static_cast<size_t>(valid_m) * valid_n * Int32Bytes;
}

size_t
MpuUnit::tileBytesForSlotKind(SlotKind kind, uint32_t valid_m,
                              uint32_t valid_n, uint32_t valid_k) const
{
    switch (kind) {
      case SlotKind::A:
        return aTileBytes(valid_m, valid_k);
      case SlotKind::B:
        return bTileBytes(valid_k, valid_n);
      case SlotKind::C:
        return cTileBytes(valid_m, valid_n);
    }

    panic("MpuUnit: unreachable slot kind");
}

size_t
MpuUnit::slotCapacityBytes(int slot_index) const
{
    switch (slot_index) {
      case SlotA0:
      case SlotA1:
        return aTileBytes(arrayRows, arrayKDepth) + (arrayRows - 1) * Int8Bytes;
      case SlotB0:
      case SlotB1:
        return bTileBytes(arrayKDepth, arrayCols) + (arrayKDepth - 1) * Int8Bytes;
      case SlotC0:
      case SlotC1:
        return cTileBytes(arrayRows, arrayCols) + (arrayRows - 1) * Int32Bytes;
    }

    panic("MpuUnit: unreachable slot capacity query");
}

Tick
MpuUnit::ticksForCycles(uint64_t cycles) const
{
    return cycles == 0 ? 0 : cycles * clockPeriod();
}

Tick
MpuUnit::bandwidthLatency(size_t bytes, uint32_t bytes_per_cycle) const
{
    const uint64_t cycles = (bytes + bytes_per_cycle - 1) / bytes_per_cycle;
    return ticksForCycles(cycles);
}

Tick
MpuUnit::matmulLatency(uint32_t valid_k) const
{
    return arrayFillLatency +
           (static_cast<Tick>(valid_k) * arraySteadyPerK) +
           arrayDrainLatency;
}

Tick
MpuUnit::matmulAccLatency(uint32_t valid_k) const
{
    return cReadBaseLatency + matmulLatency(valid_k) + cWriteBaseLatency;
}

void
MpuUnit::setSlotShape(SlotState &slot, const ParsedCmd &cmd)
{
    slot.shapeM = cmd.validM;
    slot.shapeN = cmd.validN;
    slot.shapeK = cmd.validK;
}

void
MpuUnit::clearSlotBusy(int slot_index)
{
    if (slot_index >= 0) {
        slotByIndex(slot_index).busy = false;
    }
}

void
MpuUnit::markBusyForFineCommand(const ParsedCmd &cmd)
{
    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::Load:
        slotByIndex(slotIndexForAddr(cmd.localAddrA)).busy = true;
        return;
      case Mode::Compute:
        slotByIndex(slotIndexForAddr(cmd.localAddrA)).busy = true;
        slotByIndex(slotIndexForAddr(cmd.localAddrB)).busy = true;
        slotByIndex(slotIndexForAddr(cmd.localAddrC)).busy = true;
        return;
      case Mode::Store:
        slotByIndex(slotIndexForAddr(cmd.localAddrC)).busy = true;
        return;
      case Mode::TensorLoop:
        return;
    }

    panic("MpuUnit: unreachable busy marker");
}

void
MpuUnit::clearBusyForFineCommand(const ParsedCmd &cmd)
{
    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::Load:
        clearSlotBusy(slotIndexForAddr(cmd.localAddrA));
        return;
      case Mode::Compute:
        clearSlotBusy(slotIndexForAddr(cmd.localAddrA));
        clearSlotBusy(slotIndexForAddr(cmd.localAddrB));
        clearSlotBusy(slotIndexForAddr(cmd.localAddrC));
        return;
      case Mode::Store:
        clearSlotBusy(slotIndexForAddr(cmd.localAddrC));
        return;
      case Mode::TensorLoop:
        return;
    }

    panic("MpuUnit: unreachable busy clearer");
}

void
MpuUnit::writeInt32(std::vector<uint8_t> &bytes, size_t element_index,
                    int32_t value) const
{
    const size_t offset = element_index * sizeof(value);
    panic_if(offset + sizeof(value) > bytes.size(),
             "MpuUnit: int32 write offset %zu out of range", offset);
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

int32_t
MpuUnit::readInt32(const std::vector<uint8_t> &bytes, size_t element_index) const
{
    const size_t offset = element_index * sizeof(int32_t);
    panic_if(offset + sizeof(int32_t) > bytes.size(),
             "MpuUnit: int32 read offset %zu out of range", offset);

    int32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void
MpuUnit::runMatmul(const std::vector<uint8_t> &a_bytes,
                   const std::vector<uint8_t> &b_bytes,
                   std::vector<uint8_t> &c_bytes, uint32_t valid_m,
                   uint32_t valid_n, uint32_t valid_k, bool accumulate) const
{
    c_bytes.resize(cTileBytes(valid_m, valid_n), 0);

    for (uint32_t m = 0; m < valid_m; ++m) {
        for (uint32_t n = 0; n < valid_n; ++n) {
            int32_t value = accumulate ?
                readInt32(c_bytes, static_cast<size_t>(m) * valid_n + n) : 0;
            for (uint32_t k = 0; k < valid_k; ++k) {
                const int8_t a = static_cast<int8_t>(
                    a_bytes[static_cast<size_t>(m) * valid_k + k]);
                const int8_t b = static_cast<int8_t>(
                    b_bytes[static_cast<size_t>(k) * valid_n + n]);
                value += static_cast<int32_t>(a) * static_cast<int32_t>(b);
            }
            writeInt32(c_bytes, static_cast<size_t>(m) * valid_n + n, value);
        }
    }
}

MpuUnit::LocalView
MpuUnit::makeLocalView(int slot_index, size_t offset_bytes,
                       uint32_t layout_mode, uint32_t valid_m,
                       uint32_t valid_n, uint32_t valid_k) const
{
    panic_if(layout_mode > static_cast<uint32_t>(LayoutMode::Skewed),
             "MpuUnit: unsupported layout_mode=%u", layout_mode);

    const SlotState &slot = slotByIndex(slot_index);
    LocalView view;
    view.slotIndex = slot_index;
    view.kind = slot.kind;
    view.offsetBytes = offset_bytes;
    view.layoutMode = layout_mode;
    view.validM = valid_m;
    view.validN = valid_n;
    view.validK = valid_k;

    switch (slot.kind) {
      case SlotKind::A:
        view.elemBytes = Int8Bytes;
        view.rows = valid_m;
        view.cols = valid_k;
        break;
      case SlotKind::B:
        view.elemBytes = Int8Bytes;
        view.rows = valid_k;
        view.cols = valid_n;
        break;
      case SlotKind::C:
        view.elemBytes = Int32Bytes;
        view.rows = valid_m;
        view.cols = valid_n;
        panic_if((offset_bytes % Int32Bytes) != 0,
                 "MpuUnit: C-slot local offset %zu must be %zu-byte aligned",
                 offset_bytes, Int32Bytes);
        break;
    }

    view.linearBytes = view.rows * view.cols * view.elemBytes;
    view.rowStrideBytes = view.cols * view.elemBytes;
    if (layout_mode == static_cast<uint32_t>(LayoutMode::Skewed) &&
        view.rows > 1) {
        view.rowStrideBytes += view.elemBytes;
    }
    view.physicalBytes = view.rows == 0 ? 0 :
        ((view.rows - 1) * view.rowStrideBytes) +
        (view.cols * view.elemBytes);

    panic_if(offset_bytes + view.physicalBytes > slot.bytes.size(),
             "MpuUnit: local view offset=%zu physical=%zu exceeds slot "
             "capacity=%zu", offset_bytes, view.physicalBytes,
             slot.bytes.size());
    return view;
}

MpuUnit::LocalView
MpuUnit::loadViewForSlot(int slot_index, uint32_t layout_mode,
                         size_t offset_bytes, uint32_t valid_m,
                         uint32_t valid_n, uint32_t valid_k) const
{
    return makeLocalView(slot_index, offset_bytes, layout_mode, valid_m,
                         valid_n, valid_k);
}

size_t
MpuUnit::translateLocalOffset(const LocalView &view, size_t row,
                              size_t col) const
{
    return view.offsetBytes + (row * view.rowStrideBytes) +
           (col * view.elemBytes);
}

std::vector<uint8_t>
MpuUnit::readLinearFromSlot(const LocalView &view) const
{
    const SlotState &slot = slotByIndex(view.slotIndex);
    std::vector<uint8_t> linear(view.linearBytes, 0);
    size_t linear_offset = 0;

    for (size_t row = 0; row < view.rows; ++row) {
        for (size_t col = 0; col < view.cols; ++col) {
            const size_t phys = translateLocalOffset(view, row, col);
            std::memcpy(linear.data() + linear_offset,
                        slot.bytes.data() + phys, view.elemBytes);
            linear_offset += view.elemBytes;
        }
    }

    return linear;
}

void
MpuUnit::writeLinearToSlot(const LocalView &view,
                           const std::vector<uint8_t> &linear_bytes)
{
    panic_if(linear_bytes.size() != view.linearBytes,
             "MpuUnit: local write size=%zu expected=%zu",
             linear_bytes.size(), view.linearBytes);
    SlotState &slot = slotByIndex(view.slotIndex);
    if (view.physicalBytes != 0) {
        std::fill(slot.bytes.begin() + view.offsetBytes,
                  slot.bytes.begin() + view.offsetBytes + view.physicalBytes,
                  0);
    }

    size_t linear_offset = 0;
    for (size_t row = 0; row < view.rows; ++row) {
        for (size_t col = 0; col < view.cols; ++col) {
            const size_t phys = translateLocalOffset(view, row, col);
            std::memcpy(slot.bytes.data() + phys,
                        linear_bytes.data() + linear_offset,
                        view.elemBytes);
            linear_offset += view.elemBytes;
        }
    }
}

Tick
MpuUnit::localAccessCycles(const LocalView &view, Tick *stall_out) const
{
    std::vector<uint64_t> bank_counts(localBankCount, 0);
    std::unordered_set<size_t> seen_granules;

    for (size_t row = 0; row < view.rows; ++row) {
        for (size_t col = 0; col < view.cols; ++col) {
            const size_t phys = translateLocalOffset(view, row, col);
            for (size_t byte = 0; byte < view.elemBytes; ++byte) {
                const size_t granule = (phys + byte) / localBankGranularityBytes;
                if (!seen_granules.insert(granule).second) {
                    continue;
                }
                const size_t bank = granule % localBankCount;
                bank_counts[bank]++;
            }
        }
    }

    uint64_t max_granules = 0;
    for (const uint64_t count : bank_counts) {
        max_granules = std::max(max_granules, count);
    }

    const Tick total = ticksForCycles(max_granules * localBankServiceCycles);
    const Tick stall = max_granules > 0 ?
        ticksForCycles((max_granules - 1) * localBankServiceCycles) : 0;
    if (stall_out != nullptr) {
        *stall_out = stall;
    }
    return total;
}

Tick
MpuUnit::loadLatencyForView(const LocalView &view, Tick *slot_stall) const
{
    Tick local_stall = 0;
    const Tick local = localAccessCycles(view, &local_stall);
    if (slot_stall != nullptr) {
        *slot_stall = local_stall;
    }
    return loadBaseLatency +
           bandwidthLatency(view.linearBytes, loadBandwidthBytesPerCycle) +
           local;
}

Tick
MpuUnit::storeLatencyForView(const LocalView &view, Tick *slot_stall) const
{
    Tick local_stall = 0;
    const Tick local = localAccessCycles(view, &local_stall);
    if (slot_stall != nullptr) {
        *slot_stall = local_stall;
    }
    return storeBaseLatency +
           bandwidthLatency(view.linearBytes, storeBandwidthBytesPerCycle) +
           local;
}

Tick
MpuUnit::matmulLatencyForViews(const LocalView &view_a,
                               const LocalView &view_b,
                               const LocalView &view_c,
                               uint32_t valid_k,
                               bool accumulate,
                               Tick *slot_stall) const
{
    Tick stall_a = 0;
    Tick stall_b = 0;
    Tick stall_c_read = 0;
    Tick stall_c_write = 0;
    const Tick local_a = localAccessCycles(view_a, &stall_a);
    const Tick local_b = localAccessCycles(view_b, &stall_b);
    const Tick local_c_read =
        accumulate ? localAccessCycles(view_c, &stall_c_read) : 0;
    const Tick local_c_write = localAccessCycles(view_c, &stall_c_write);
    const Tick total = local_a + local_b + local_c_write +
        (accumulate ? (cReadBaseLatency + local_c_read + cWriteBaseLatency) : 0) +
        matmulLatency(valid_k);
    if (slot_stall != nullptr) {
        *slot_stall = stall_a + stall_b + stall_c_read + stall_c_write;
    }
    return total;
}

uint64_t
MpuUnit::outputTileKey(uint32_t m_index, uint32_t n_index) const
{
    return (static_cast<uint64_t>(m_index) << 32) | n_index;
}

void
MpuUnit::updateSlotForTensorLoad(int slot_index, size_t offset_bytes,
                                 uint32_t layout_mode,
                                 const std::vector<uint8_t> &linear_bytes,
                                 uint32_t valid_m, uint32_t valid_n,
                                 uint32_t valid_k, bool dirty)
{
    const LocalView view = makeLocalView(slot_index, offset_bytes, layout_mode,
                                         valid_m, valid_n, valid_k);
    SlotState &slot = slotByIndex(slot_index);
    writeLinearToSlot(view, linear_bytes);
    slot.valid = true;
    slot.busy = false;
    slot.dirty = dirty;
    slot.shapeM = valid_m;
    slot.shapeN = valid_n;
    slot.shapeK = valid_k;
    slot.layoutMode = layout_mode;
}

void
MpuUnit::validateLoadCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    const int slot_index = slotIndexForAddr(cmd.localAddrA);
    panic_if(slot_index < 0,
             "MpuUnit: invalid load destination local address %#x",
             cmd.localAddrA);
    for (size_t i = 7; i < cmd.words.size(); ++i) {
        panic_if(cmd.words[i] != 0,
                 "MpuUnit: load reserved Word %zu must be zero, got %#x",
                 i, cmd.words[i]);
    }

    const SlotState &slot = slotByIndex(slot_index);
    const LocalView view = loadViewForSlot(
        slot_index, cmd.layoutMode, localOffsetBytes(cmd.localAddrA),
        cmd.validM, cmd.validN, cmd.validK);
    validateSpmAddress(cmd.spmAddrA, view.linearBytes, "load source");
    panic_if(slot.busy, "MpuUnit: load destination slot %#x is busy",
             cmd.localAddrA);
}

void
MpuUnit::validateComputeCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    validateSlotAddress(cmd.localAddrA, SlotKind::A, "compute src0");
    validateSlotAddress(cmd.localAddrB, SlotKind::B, "compute src1");
    validateSlotAddress(cmd.localAddrC, SlotKind::C, "compute dst");
    panic_if(cmd.computeModeFlags != 0,
             "MpuUnit: compute_mode_flags must be zero in v1, got %#x",
             cmd.computeModeFlags);
    for (size_t i = 9; i < cmd.words.size(); ++i) {
        panic_if(cmd.words[i] != 0,
                 "MpuUnit: compute reserved Word %zu must be zero, got %#x",
                 i, cmd.words[i]);
    }
    panic_if(cmd.subop != ComputeSubop::Matmul &&
             cmd.subop != ComputeSubop::MatmulAcc,
             "MpuUnit: unsupported compute subop=%u", cmd.words[7]);

    const SlotState &slot_a = slotByIndex(slotIndexForAddr(cmd.localAddrA));
    const SlotState &slot_b = slotByIndex(slotIndexForAddr(cmd.localAddrB));
    const SlotState &slot_c = slotByIndex(slotIndexForAddr(cmd.localAddrC));
    panic_if(!slot_a.valid, "MpuUnit: compute A slot is invalid");
    panic_if(!slot_b.valid, "MpuUnit: compute B slot is invalid");
    panic_if(slot_a.busy || slot_b.busy || slot_c.busy,
             "MpuUnit: compute slot is busy");
    validateSlotShape(slot_a, cmd, "A");
    validateSlotShape(slot_b, cmd, "B");
    const uint32_t c_layout = slot_c.valid ? slot_c.layoutMode :
        static_cast<uint32_t>(LayoutMode::Normal);
    makeLocalView(slotIndexForAddr(cmd.localAddrA), localOffsetBytes(cmd.localAddrA),
                  slot_a.layoutMode, cmd.validM, cmd.validN, cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrB), localOffsetBytes(cmd.localAddrB),
                  slot_b.layoutMode, cmd.validM, cmd.validN, cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrC), localOffsetBytes(cmd.localAddrC),
                  c_layout, cmd.validM, cmd.validN, cmd.validK);
    if (cmd.subop == ComputeSubop::MatmulAcc) {
        panic_if(!slot_c.valid, "MpuUnit: MATMUL_ACC requires valid C slot");
        validateSlotShape(slot_c, cmd, "C");
    }
}

void
MpuUnit::validateStoreCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    validateSlotAddress(cmd.localAddrC, SlotKind::C, "store source");
    for (size_t i = 7; i < cmd.words.size(); ++i) {
        panic_if(cmd.words[i] != 0,
                 "MpuUnit: store reserved Word %zu must be zero, got %#x",
                 i, cmd.words[i]);
    }

    const SlotState &slot = slotByIndex(slotIndexForAddr(cmd.localAddrC));
    panic_if(!slot.valid, "MpuUnit: store source C slot is invalid");
    panic_if(slot.busy, "MpuUnit: store source C slot is busy");
    validateSlotShape(slot, cmd, "store source");
    panic_if(slot.layoutMode != cmd.layoutMode,
             "MpuUnit: store layout_mode=%u does not match slot layout=%u",
             cmd.layoutMode, slot.layoutMode);
    const LocalView view = makeLocalView(slotIndexForAddr(cmd.localAddrC),
                                         localOffsetBytes(cmd.localAddrC),
                                         cmd.layoutMode, cmd.validM,
                                         cmd.validN, cmd.validK);
    validateSpmAddress(cmd.spmAddrC, view.linearBytes, "store destination");
}

void
MpuUnit::validateTensorLoopCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    validateSlotAddress(cmd.localAddrA, SlotKind::A, "tensor_loop base A");
    validateSlotAddress(cmd.localAddrB, SlotKind::B, "tensor_loop base B");
    validateSlotAddress(cmd.localAddrC, SlotKind::C, "tensor_loop base C");
    panic_if(!spmContains(cmd.spmAddrA, 1) || !spmContains(cmd.spmAddrB, 1) ||
             !spmContains(cmd.spmAddrC, 1),
             "MpuUnit: tensor_loop base SPM address is invalid");
    panic_if(cmd.outerCount == 0 || cmd.innerCount == 0,
             "MpuUnit: tensor_loop counts must be greater than zero");
    panic_if(static_cast<uint32_t>(cmd.outerAxis) >
                 static_cast<uint32_t>(Axis::K) ||
             static_cast<uint32_t>(cmd.innerAxis) >
                 static_cast<uint32_t>(Axis::K),
             "MpuUnit: tensor_loop axis must be one of M/N/K");
    panic_if(cmd.outerAxis == cmd.innerAxis,
             "MpuUnit: tensor_loop outer_axis must differ from inner_axis");
    panic_if(cmd.subop != ComputeSubop::Matmul &&
             cmd.subop != ComputeSubop::MatmulAcc,
             "MpuUnit: tensor_loop subop must be MATMUL or MATMUL_ACC");
    panic_if(cmd.subop == ComputeSubop::MatmulAcc && !cmd.autoLoadCForAcc,
             "MpuUnit: tensor_loop MATMUL_ACC requires auto_load_c_for_acc=1");
    panic_if((cmd.words[15] & TensorStepCfgReservedMask) != 0,
             "MpuUnit: tensor_loop reserved step_cfg bits must be zero");

    makeLocalView(slotIndexForAddr(cmd.localAddrA), localOffsetBytes(cmd.localAddrA),
                  cmd.tensorLayoutA, cmd.validM, cmd.validN, cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrB), localOffsetBytes(cmd.localAddrB),
                  cmd.tensorLayoutB, cmd.validM, cmd.validN, cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrC), localOffsetBytes(cmd.localAddrC),
                  cmd.tensorLayoutC, cmd.validM, cmd.validN, cmd.validK);
    if (cmd.pingpongA) {
        makeLocalView(alternateSlotIndex(slotIndexForAddr(cmd.localAddrA)),
                      localOffsetBytes(cmd.localAddrA), cmd.tensorLayoutA,
                      cmd.validM, cmd.validN, cmd.validK);
    }
    if (cmd.pingpongB) {
        makeLocalView(alternateSlotIndex(slotIndexForAddr(cmd.localAddrB)),
                      localOffsetBytes(cmd.localAddrB), cmd.tensorLayoutB,
                      cmd.validM, cmd.validN, cmd.validK);
    }
    if (cmd.pingpongC) {
        makeLocalView(alternateSlotIndex(slotIndexForAddr(cmd.localAddrC)),
                      localOffsetBytes(cmd.localAddrC), cmd.tensorLayoutC,
                      cmd.validM, cmd.validN, cmd.validK);
    }
}

void
MpuUnit::validateParsedCommand(const ParsedCmd &cmd) const
{
    panic_if((cmd.words[0] & 0x3fU) != 0,
             "MpuUnit: Common_Header reserved bits must be zero, got %#x",
             cmd.words[0] & 0x3fU);
    panic_if(cmd.dataType != 0,
             "MpuUnit: unsupported data_type=%u", cmd.dataType);

    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::Load:
        validateLoadCommand(cmd);
        return;
      case Mode::Compute:
        validateComputeCommand(cmd);
        return;
      case Mode::Store:
        validateStoreCommand(cmd);
        return;
      case Mode::TensorLoop:
        validateTensorLoopCommand(cmd);
        return;
    }

    panic("MpuUnit: unsupported mode=%u", cmd.mode);
}

Addr
MpuUnit::axisDeltaA(Axis axis, uint32_t valid_m, uint32_t valid_n,
                    uint32_t valid_k) const
{
    (void)valid_n;
    switch (axis) {
      case Axis::M:
      case Axis::K:
        return aTileBytes(valid_m, valid_k);
      case Axis::N:
        return 0;
    }

    panic("MpuUnit: unreachable A axis delta");
}

Addr
MpuUnit::axisDeltaB(Axis axis, uint32_t valid_m, uint32_t valid_n,
                    uint32_t valid_k) const
{
    (void)valid_m;
    switch (axis) {
      case Axis::N:
      case Axis::K:
        return bTileBytes(valid_k, valid_n);
      case Axis::M:
        return 0;
    }

    panic("MpuUnit: unreachable B axis delta");
}

Addr
MpuUnit::axisDeltaC(Axis axis, uint32_t valid_m, uint32_t valid_n,
                    uint32_t valid_k) const
{
    (void)valid_k;
    switch (axis) {
      case Axis::M:
      case Axis::N:
        return cTileBytes(valid_m, valid_n);
      case Axis::K:
        return 0;
    }

    panic("MpuUnit: unreachable C axis delta");
}

void
MpuUnit::buildTensorLoopPlans()
{
    const size_t a_bytes = aTileBytes(parsedCmd.validM, parsedCmd.validK);
    const size_t b_bytes = bTileBytes(parsedCmd.validK, parsedCmd.validN);
    const size_t c_bytes = cTileBytes(parsedCmd.validM, parsedCmd.validN);
    const bool includes_k =
        parsedCmd.outerAxis == Axis::K || parsedCmd.innerAxis == Axis::K;
    const uint32_t k_total = !includes_k ? 1U :
        (parsedCmd.outerAxis == Axis::K ? parsedCmd.outerCount :
                                         parsedCmd.innerCount);
    std::unordered_map<uint64_t, int> output_to_c_slot;
    uint64_t output_tile_ordinal = 0;

    Addr outer_a = parsedCmd.spmAddrA;
    Addr outer_b = parsedCmd.spmAddrB;
    Addr outer_c = parsedCmd.spmAddrC;

    for (uint32_t outer = 0; outer < parsedCmd.outerCount; ++outer) {
        Addr inner_a = outer_a;
        Addr inner_b = outer_b;
        Addr inner_c = outer_c;

        for (uint32_t inner = 0; inner < parsedCmd.innerCount; ++inner) {
            uint32_t m_index = 0;
            uint32_t n_index = 0;
            uint32_t k_index = 0;

            const auto apply_axis = [&](Axis axis, uint32_t value) {
                switch (axis) {
                  case Axis::M:
                    m_index = value;
                    break;
                  case Axis::N:
                    n_index = value;
                    break;
                  case Axis::K:
                    k_index = value;
                    break;
                }
            };
            apply_axis(parsedCmd.outerAxis, outer);
            apply_axis(parsedCmd.innerAxis, inner);

            validateSpmAddress(inner_a, a_bytes, "tensor_loop A tile");
            validateSpmAddress(inner_b, b_bytes, "tensor_loop B tile");
            validateSpmAddress(inner_c, c_bytes, "tensor_loop C tile");

            const uint64_t out_key = outputTileKey(m_index, n_index);
            auto c_it = output_to_c_slot.find(out_key);
            if (c_it == output_to_c_slot.end()) {
                int slot_c = slotIndexForAddr(parsedCmd.localAddrC);
                if (parsedCmd.pingpongC && (output_tile_ordinal % 2U) == 1U) {
                    slot_c = alternateSlotIndex(slot_c);
                }
                c_it = output_to_c_slot.emplace(out_key, slot_c).first;
                output_tile_ordinal++;
            }

            const bool first_k = !includes_k || (k_index == 0U);
            const bool last_k = !includes_k || (k_index + 1U == k_total);
            const uint64_t tile_ordinal = iterationPlans.size();
            int slot_a = slotIndexForAddr(parsedCmd.localAddrA);
            int slot_b = slotIndexForAddr(parsedCmd.localAddrB);
            if (parsedCmd.pingpongA && ((tile_ordinal & 0x1U) != 0)) {
                slot_a = alternateSlotIndex(slot_a);
            }
            if (parsedCmd.pingpongB && ((tile_ordinal & 0x1U) != 0)) {
                slot_b = alternateSlotIndex(slot_b);
            }

            IterationPlan plan;
            plan.tensorLoop = true;
            plan.validM = parsedCmd.validM;
            plan.validN = parsedCmd.validN;
            plan.validK = parsedCmd.validK;
            plan.tensorSlotA = slot_a;
            plan.tensorSlotB = slot_b;
            plan.tensorSlotC = c_it->second;
            plan.tensorOffsetA = localOffsetBytes(parsedCmd.localAddrA);
            plan.tensorOffsetB = localOffsetBytes(parsedCmd.localAddrB);
            plan.tensorOffsetC = localOffsetBytes(parsedCmd.localAddrC);
            plan.tensorLayoutA = parsedCmd.tensorLayoutA;
            plan.tensorLayoutB = parsedCmd.tensorLayoutB;
            plan.tensorLayoutC = parsedCmd.tensorLayoutC;
            plan.tensorAAddr = inner_a;
            plan.tensorBAddr = inner_b;
            plan.tensorCAddr = inner_c;
            plan.firstK = first_k;
            plan.lastK = last_k;
            plan.kExpanded = includes_k;
            plan.outputTileKey = out_key;
            plan.doLoadA = true;
            plan.doLoadB = true;
            plan.doLoadCOld = parsedCmd.subop == ComputeSubop::MatmulAcc &&
                parsedCmd.autoLoadCForAcc && (!includes_k || first_k);
            plan.doStoreC = !includes_k || last_k;
            plan.subop = includes_k ? (first_k ? ComputeSubop::Matmul :
                                                 ComputeSubop::MatmulAcc)
                                    : parsedCmd.subop;
            plan.tensorA.assign(a_bytes, 0);
            plan.tensorB.assign(b_bytes, 0);
            plan.tensorCOld.assign(c_bytes, 0);
            plan.tensorCResult.assign(c_bytes, 0);

            const LocalView view_a = makeLocalView(
                plan.tensorSlotA, plan.tensorOffsetA, plan.tensorLayoutA,
                plan.validM, plan.validN, plan.validK);
            const LocalView view_b = makeLocalView(
                plan.tensorSlotB, plan.tensorOffsetB, plan.tensorLayoutB,
                plan.validM, plan.validN, plan.validK);
            const LocalView view_c = makeLocalView(
                plan.tensorSlotC, plan.tensorOffsetC, plan.tensorLayoutC,
                plan.validM, plan.validN, plan.validK);

            Tick slot_stall = 0;
            Tick latency = 0;
            Tick spm_wait = 0;
            Tick queued_read_service = 0;
            if (plan.doLoadA) {
                Tick view_stall = 0;
                latency += loadLatencyForView(view_a, &view_stall);
                queued_read_service +=
                    bandwidthLatency(a_bytes, loadBandwidthBytesPerCycle);
            }
            if (plan.doLoadB) {
                Tick view_stall = 0;
                latency += loadLatencyForView(view_b, &view_stall);
                spm_wait += queued_read_service;
                queued_read_service +=
                    bandwidthLatency(b_bytes, loadBandwidthBytesPerCycle);
            }
            if (plan.doLoadCOld) {
                Tick view_stall = 0;
                latency += loadLatencyForView(view_c, &view_stall);
                spm_wait += queued_read_service;
                queued_read_service +=
                    bandwidthLatency(c_bytes, loadBandwidthBytesPerCycle);
            }
            latency += matmulLatencyForViews(view_a, view_b, view_c,
                                             plan.validK,
                                             plan.subop == ComputeSubop::MatmulAcc,
                                             &slot_stall);
            if (plan.doStoreC) {
                Tick store_stall = 0;
                latency += storeLatencyForView(view_c, &store_stall);
                slot_stall += store_stall;
            }
            latency += ticksForCycles(1);

            plan.modeledSpmStall = spm_wait;
            plan.modeledSlotStall = slot_stall;
            plan.modeledLatency = latency;
            iterationPlans.push_back(std::move(plan));

            inner_a += parsedCmd.innerStepTiles *
                       axisDeltaA(parsedCmd.innerAxis, parsedCmd.validM,
                                  parsedCmd.validN, parsedCmd.validK);
            inner_b += parsedCmd.innerStepTiles *
                       axisDeltaB(parsedCmd.innerAxis, parsedCmd.validM,
                                  parsedCmd.validN, parsedCmd.validK);
            inner_c += parsedCmd.innerStepTiles *
                       axisDeltaC(parsedCmd.innerAxis, parsedCmd.validM,
                                  parsedCmd.validN, parsedCmd.validK);
        }

        outer_a += parsedCmd.outerStepTiles *
                   axisDeltaA(parsedCmd.outerAxis, parsedCmd.validM,
                              parsedCmd.validN, parsedCmd.validK);
        outer_b += parsedCmd.outerStepTiles *
                   axisDeltaB(parsedCmd.outerAxis, parsedCmd.validM,
                              parsedCmd.validN, parsedCmd.validK);
        outer_c += parsedCmd.outerStepTiles *
                   axisDeltaC(parsedCmd.outerAxis, parsedCmd.validM,
                              parsedCmd.validN, parsedCmd.validK);
    }
}

void
MpuUnit::buildIterationPlans(ActiveExecution &exec)
{
    iterationPlans.clear();

    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Load: {
        IterationPlan plan;
        plan.validM = parsedCmd.validM;
        plan.validN = parsedCmd.validN;
        plan.validK = parsedCmd.validK;
        plan.loadSlotIndex = slotIndexForAddr(parsedCmd.localAddrA);
        plan.loadOffset = localOffsetBytes(parsedCmd.localAddrA);
        plan.loadLayoutMode = parsedCmd.layoutMode;
        plan.loadSpmAddr = parsedCmd.spmAddrA;
        {
            Tick slot_stall = 0;
            const LocalView view = loadViewForSlot(plan.loadSlotIndex,
                                                   plan.loadLayoutMode,
                                                   plan.loadOffset,
                                                   plan.validM,
                                                   plan.validN,
                                                   plan.validK);
            plan.modeledLatency = loadLatencyForView(view, &slot_stall);
            plan.modeledSlotStall = slot_stall;
        }
        iterationPlans.push_back(std::move(plan));
        exec.readMask = 1U;
        exec.writeMask = 0U;
        exec.repetition = 1;
        break;
      }
      case Mode::Compute: {
        IterationPlan plan;
        plan.validM = parsedCmd.validM;
        plan.validN = parsedCmd.validN;
        plan.validK = parsedCmd.validK;
        plan.computeSlotA = slotIndexForAddr(parsedCmd.localAddrA);
        plan.computeSlotB = slotIndexForAddr(parsedCmd.localAddrB);
        plan.computeSlotC = slotIndexForAddr(parsedCmd.localAddrC);
        plan.computeOffsetA = localOffsetBytes(parsedCmd.localAddrA);
        plan.computeOffsetB = localOffsetBytes(parsedCmd.localAddrB);
        plan.computeOffsetC = localOffsetBytes(parsedCmd.localAddrC);
        plan.subop = parsedCmd.subop;
        {
            const SlotState &slot_a = slotByIndex(plan.computeSlotA);
            const SlotState &slot_b = slotByIndex(plan.computeSlotB);
            const SlotState &slot_c = slotByIndex(plan.computeSlotC);
            const uint32_t c_layout = slot_c.valid ? slot_c.layoutMode :
                static_cast<uint32_t>(LayoutMode::Normal);
            const LocalView view_a = makeLocalView(plan.computeSlotA,
                                                   plan.computeOffsetA,
                                                   slot_a.layoutMode,
                                                   plan.validM, plan.validN,
                                                   plan.validK);
            const LocalView view_b = makeLocalView(plan.computeSlotB,
                                                   plan.computeOffsetB,
                                                   slot_b.layoutMode,
                                                   plan.validM, plan.validN,
                                                   plan.validK);
            const LocalView view_c = makeLocalView(plan.computeSlotC,
                                                   plan.computeOffsetC,
                                                   c_layout,
                                                   plan.validM, plan.validN,
                                                   plan.validK);
            Tick slot_stall = 0;
            plan.modeledLatency = matmulLatencyForViews(
                view_a, view_b, view_c, plan.validK,
                plan.subop == ComputeSubop::MatmulAcc, &slot_stall);
            plan.modeledSlotStall = slot_stall;
        }
        iterationPlans.push_back(std::move(plan));
        exec.readMask = 0U;
        exec.writeMask = 0U;
        exec.repetition = 1;
        break;
      }
      case Mode::Store: {
        IterationPlan plan;
        plan.validM = parsedCmd.validM;
        plan.validN = parsedCmd.validN;
        plan.validK = parsedCmd.validK;
        plan.storeSlotIndex = slotIndexForAddr(parsedCmd.localAddrC);
        plan.storeOffset = localOffsetBytes(parsedCmd.localAddrC);
        plan.storeLayoutMode = parsedCmd.layoutMode;
        plan.storeSpmAddr = parsedCmd.spmAddrC;
        {
            Tick slot_stall = 0;
            const LocalView view = makeLocalView(plan.storeSlotIndex,
                                                 plan.storeOffset,
                                                 plan.storeLayoutMode,
                                                 plan.validM, plan.validN,
                                                 plan.validK);
            plan.modeledLatency = storeLatencyForView(view, &slot_stall);
            plan.modeledSlotStall = slot_stall;
        }
        iterationPlans.push_back(std::move(plan));
        exec.readMask = 0U;
        exec.writeMask = 1U;
        exec.repetition = 1;
        break;
      }
      case Mode::TensorLoop:
        buildTensorLoopPlans();
        exec.readMask = 1U;
        exec.writeMask = 1U;
        exec.repetition = iterationPlans.size();
        break;
    }
}

void
MpuUnit::updateTensorLoopStats(const IterationPlan &plan)
{
    totalInternalLoadsValue +=
        (plan.doLoadA ? 1U : 0U) + (plan.doLoadB ? 1U : 0U) +
        (plan.doLoadCOld ? 1U : 0U);
    totalInternalComputesValue += 1U;
    totalInternalStoresValue += plan.doStoreC ? 1U : 0U;
    totalTilesValue += 1U;
    totalAccTilesValue += plan.subop == ComputeSubop::MatmulAcc ? 1U : 0U;
    stallCyclesWaitingForSPMValue += plan.modeledSpmStall;
    stallCyclesWaitingForSlotValue += plan.modeledSlotStall;
    computedTotalLatencyValue += plan.modeledLatency;
}

MpuUnit::IterationPlan &
MpuUnit::iterationPlan(uint64_t iteration)
{
    panic_if(iteration >= iterationPlans.size(),
             "%s: iteration %llu out of range (plans=%llu)", name(),
             static_cast<unsigned long long>(iteration),
             static_cast<unsigned long long>(iterationPlans.size()));
    return iterationPlans.at(iteration);
}

const MpuUnit::IterationPlan *
MpuUnit::findIterationPlan(uint64_t iteration) const
{
    if (iteration >= iterationPlans.size()) {
        return nullptr;
    }

    return &iterationPlans.at(iteration);
}

void
MpuUnit::startExecuteCommand(const std::vector<uint8_t> &cmd)
{
    const uint64_t total_prologues = activeExecution.prologueCount;
    const uint64_t total_executes = activeExecution.executeCount;
    const uint64_t total_epilogues = activeExecution.epilogueCount;
    const uint64_t total_reads = activeExecution.completedReadRespCount;
    const uint64_t total_writes = activeExecution.completedWriteRespCount;
    const uint64_t total_iterations = activeExecution.completedIterations;

    activeExecution = ActiveExecution{};
    activeExecution.cmd = cmd;
    activeExecution.fields = parseCmdFields(extractCmdWord(cmd));
    activeExecution.phase = Phase::Prologue;
    activeExecution.prologueCount = total_prologues;
    activeExecution.executeCount = total_executes;
    activeExecution.epilogueCount = total_epilogues;
    activeExecution.completedReadRespCount = total_reads;
    activeExecution.completedWriteRespCount = total_writes;
    activeExecution.completedIterations = total_iterations;
    activeExecution.readMask = 0;
    activeExecution.writeMask = 0;
    activeExecution.repetition = 1;
    activeExecution.reserved = 0;
    activeExecution.nextIterationToPrepare = 0;
    activeExecution.nextIterationToRetire = 0;
    activeExecution.completionIssued = false;
    activeExecution.finalizePending = false;

    for (PortID port = 0; port < static_cast<PortID>(memSidePorts.size());
         ++port) {
        loadQueues[port].clear();
        storeQueues[port].clear();
        memPortBusy[port] = false;
    }
    execQueue.clear();
    activeExecOp.reset();
    iterationStates.clear();
    maxConcurrentMicroOps = 0;

    onCommandBegin(activeExecution);
    prepareIteration(activeExecution, 0);
}

void
MpuUnit::onCommandBegin(ActiveExecution &exec)
{
    resetCommandState();
    parsedCmd = parseCommand(exec.cmd);
    parsedCmdValid = true;
    validateParsedCommand(parsedCmd);

    buildIterationPlans(exec);

    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Load:
        loadCount++;
        markBusyForFineCommand(parsedCmd);
        totalInternalLoadsValue++;
        stallCyclesWaitingForSPMValue += iterationPlans[0].modeledSpmStall;
        stallCyclesWaitingForSlotValue += iterationPlans[0].modeledSlotStall;
        computedTotalLatencyValue += iterationPlans[0].modeledLatency;
        break;
      case Mode::Compute:
        computeCount++;
        markBusyForFineCommand(parsedCmd);
        totalInternalComputesValue++;
        totalTilesValue++;
        if (parsedCmd.subop == ComputeSubop::MatmulAcc) {
            totalAccTilesValue++;
        }
        stallCyclesWaitingForSPMValue += iterationPlans[0].modeledSpmStall;
        stallCyclesWaitingForSlotValue += iterationPlans[0].modeledSlotStall;
        computedTotalLatencyValue += iterationPlans[0].modeledLatency;
        break;
      case Mode::Store:
        storeCount++;
        markBusyForFineCommand(parsedCmd);
        totalInternalStoresValue++;
        stallCyclesWaitingForSPMValue += iterationPlans[0].modeledSpmStall;
        stallCyclesWaitingForSlotValue += iterationPlans[0].modeledSlotStall;
        computedTotalLatencyValue += iterationPlans[0].modeledLatency;
        break;
      case Mode::TensorLoop:
        tensorLoopCount++;
        tensorLoopExpandedTilesCount += iterationPlans.size();
        for (const auto &plan : iterationPlans) {
            tensorLoopExpandedAccTilesCount +=
                plan.subop == ComputeSubop::MatmulAcc ? 1U : 0U;
            updateTensorLoopStats(plan);
        }
        break;
    }
}

void
MpuUnit::buildMvinRequests(ActiveExecution &exec,
                           std::vector<MemRequestDesc> &reqs)
{
    const IterationPlan *plan = findIterationPlan(exec.iteration);
    if (plan == nullptr) {
        return;
    }

    uint64_t token = nextMemTxnToken;
    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Load: {
        const LocalView view = makeLocalView(plan->loadSlotIndex,
                                             plan->loadOffset,
                                             plan->loadLayoutMode,
                                             plan->validM,
                                             plan->validN,
                                             plan->validK);
        MemRequestDesc req;
        req.portId = 0;
        req.addr = plan->loadSpmAddr;
        req.size = view.linearBytes;
        reqs.push_back(req);
        pendingLoadTxns.emplace(token, PendingLoadTxn{
            exec.iteration, PendingLoadKind::SlotLoad, plan->loadSlotIndex});
        break;
      }
      case Mode::TensorLoop:
        if (plan->doLoadA) {
            MemRequestDesc req;
            req.portId = 0;
            req.addr = plan->tensorAAddr;
            req.size = aTileBytes(plan->validM, plan->validK);
            reqs.push_back(req);
            pendingLoadTxns.emplace(token++, PendingLoadTxn{
                exec.iteration, PendingLoadKind::TensorA, plan->tensorSlotA});
        }
        if (plan->doLoadB) {
            MemRequestDesc req;
            req.portId = 0;
            req.addr = plan->tensorBAddr;
            req.size = bTileBytes(plan->validK, plan->validN);
            reqs.push_back(req);
            pendingLoadTxns.emplace(token++, PendingLoadTxn{
                exec.iteration, PendingLoadKind::TensorB, plan->tensorSlotB});
        }
        if (plan->doLoadCOld) {
            MemRequestDesc req;
            req.portId = 0;
            req.addr = plan->tensorCAddr;
            req.size = cTileBytes(plan->validM, plan->validN);
            reqs.push_back(req);
            pendingLoadTxns.emplace(token, PendingLoadTxn{
                exec.iteration, PendingLoadKind::TensorCOld, plan->tensorSlotC});
        }
        break;
      case Mode::Compute:
      case Mode::Store:
        break;
    }
}

void
MpuUnit::onMvinResponse(ActiveExecution &exec, const MemTxnContext &txn,
                        PacketPtr pkt)
{
    auto it = pendingLoadTxns.find(txn.token);
    panic_if(it == pendingLoadTxns.end(),
             "%s: unexpected MPU load token=%llu", name(),
             static_cast<unsigned long long>(txn.token));

    const PendingLoadTxn pending = it->second;
    pendingLoadTxns.erase(it);
    const uint8_t *data = pkt->getConstPtr<uint8_t>();

    switch (pending.kind) {
      case PendingLoadKind::SlotLoad: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        const LocalView view = makeLocalView(plan.loadSlotIndex,
                                             plan.loadOffset,
                                             plan.loadLayoutMode,
                                             plan.validM,
                                             plan.validN,
                                             plan.validK);
        std::vector<uint8_t> linear(data, data + pkt->getSize());
        panic_if(linear.size() != view.linearBytes,
                 "MpuUnit: load response size=%zu expected=%zu",
                 linear.size(), view.linearBytes);
        SlotState &slot = slotByIndex(plan.loadSlotIndex);
        writeLinearToSlot(view, linear);
        slot.valid = true;
        slot.dirty = false;
        slot.busy = false;
        slot.layoutMode = plan.loadLayoutMode;
        setSlotShape(slot, parsedCmd);
        break;
      }
      case PendingLoadKind::TensorA: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        plan.tensorA.assign(data, data + pkt->getSize());
        updateSlotForTensorLoad(plan.tensorSlotA, plan.tensorOffsetA,
                                plan.tensorLayoutA, plan.tensorA,
                                plan.validM, plan.validN, plan.validK,
                                false);
        break;
      }
      case PendingLoadKind::TensorB: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        plan.tensorB.assign(data, data + pkt->getSize());
        updateSlotForTensorLoad(plan.tensorSlotB, plan.tensorOffsetB,
                                plan.tensorLayoutB, plan.tensorB,
                                plan.validM, plan.validN, plan.validK,
                                false);
        break;
      }
      case PendingLoadKind::TensorCOld: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        plan.tensorCOld.assign(data, data + pkt->getSize());
        updateSlotForTensorLoad(plan.tensorSlotC, plan.tensorOffsetC,
                                plan.tensorLayoutC, plan.tensorCOld,
                                plan.validM, plan.validN, plan.validK,
                                false);
        break;
      }
    }

    (void)exec;
}

Tick
MpuUnit::execute(ActiveExecution &exec)
{
    IterationPlan *plan_ptr = const_cast<IterationPlan *>(findIterationPlan(
        exec.iteration));
    if (plan_ptr == nullptr) {
        return 0;
    }
    IterationPlan &plan = *plan_ptr;

    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Load:
        return loadBaseLatency;
      case Mode::Compute: {
        SlotState &slot_c = slotByIndex(plan.computeSlotC);
        const SlotState &slot_a = slotByIndex(plan.computeSlotA);
        const SlotState &slot_b = slotByIndex(plan.computeSlotB);
        const uint32_t c_layout = slot_c.valid ? slot_c.layoutMode :
            static_cast<uint32_t>(LayoutMode::Normal);
        const LocalView view_a = makeLocalView(plan.computeSlotA,
                                               plan.computeOffsetA,
                                               slot_a.layoutMode,
                                               plan.validM, plan.validN,
                                               plan.validK);
        const LocalView view_b = makeLocalView(plan.computeSlotB,
                                               plan.computeOffsetB,
                                               slot_b.layoutMode,
                                               plan.validM, plan.validN,
                                               plan.validK);
        const LocalView view_c = makeLocalView(plan.computeSlotC,
                                               plan.computeOffsetC,
                                               c_layout,
                                               plan.validM, plan.validN,
                                               plan.validK);
        const std::vector<uint8_t> a_bytes = readLinearFromSlot(view_a);
        const std::vector<uint8_t> b_bytes = readLinearFromSlot(view_b);
        std::vector<uint8_t> c_bytes =
            plan.subop == ComputeSubop::MatmulAcc ?
            readLinearFromSlot(view_c) : std::vector<uint8_t>();
        runMatmul(a_bytes, b_bytes, c_bytes, plan.validM, plan.validN,
                  plan.validK, plan.subop == ComputeSubop::MatmulAcc);
        writeLinearToSlot(view_c, c_bytes);
        slot_c.valid = true;
        slot_c.dirty = true;
        slot_c.busy = false;
        slot_c.shapeM = plan.validM;
        slot_c.shapeN = plan.validN;
        slot_c.shapeK = plan.validK;
        if (!slot_c.valid) {
            slot_c.layoutMode = c_layout;
        }
        clearSlotBusy(plan.computeSlotA);
        clearSlotBusy(plan.computeSlotB);

        Tick slot_stall = 0;
        const Tick latency = matmulLatencyForViews(
            view_a, view_b, view_c, plan.validK,
            plan.subop == ComputeSubop::MatmulAcc, &slot_stall);
        if (plan.subop == ComputeSubop::Matmul) {
            matmulCountValue++;
            if (slot_c.layoutMode != c_layout) {
                slot_c.layoutMode = c_layout;
            }
            return latency;
        }

        matmulAccCountValue++;
        if (slot_c.layoutMode != c_layout) {
            slot_c.layoutMode = c_layout;
        }
        return latency;
      }
      case Mode::Store:
        return storeBaseLatency;
      case Mode::TensorLoop: {
        const LocalView view_c = makeLocalView(plan.tensorSlotC,
                                               plan.tensorOffsetC,
                                               plan.tensorLayoutC,
                                               plan.validM, plan.validN,
                                               plan.validK);
        std::vector<uint8_t> result;
        if (plan.subop == ComputeSubop::MatmulAcc) {
            auto acc_it = tensorLoopAccumulators.find(plan.outputTileKey);
            panic_if(acc_it == tensorLoopAccumulators.end(),
                     "MpuUnit: tensor_loop MATMUL_ACC missing accumulator for "
                     "tile=%llu",
                     static_cast<unsigned long long>(plan.outputTileKey));
            result = acc_it->second;
            runMatmul(plan.tensorA, plan.tensorB, result, plan.validM,
                      plan.validN, plan.validK, true);
            matmulAccCountValue++;
        } else {
            result.assign(cTileBytes(plan.validM, plan.validN), 0);
            runMatmul(plan.tensorA, plan.tensorB, result, plan.validM,
                      plan.validN, plan.validK, false);
            matmulCountValue++;
        }

        tensorLoopAccumulators[plan.outputTileKey] = result;
        plan.tensorCResult = result;
        updateSlotForTensorLoad(plan.tensorSlotC, plan.tensorOffsetC,
                                plan.tensorLayoutC, result,
                                plan.validM, plan.validN, plan.validK,
                                true);

        const LocalView view_a = makeLocalView(plan.tensorSlotA,
                                               plan.tensorOffsetA,
                                               plan.tensorLayoutA,
                                               plan.validM, plan.validN,
                                               plan.validK);
        const LocalView view_b = makeLocalView(plan.tensorSlotB,
                                               plan.tensorOffsetB,
                                               plan.tensorLayoutB,
                                               plan.validM, plan.validN,
                                               plan.validK);
        Tick slot_stall = 0;
        return matmulLatencyForViews(view_a, view_b, view_c, plan.validK,
                                     plan.subop == ComputeSubop::MatmulAcc,
                                     &slot_stall);
      }
    }

    panic("MpuUnit: unreachable execute mode");
}

void
MpuUnit::buildMvoutRequests(ActiveExecution &exec,
                            std::vector<MemRequestDesc> &reqs)
{
    const IterationPlan *plan = findIterationPlan(exec.iteration);
    if (plan == nullptr) {
        return;
    }

    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Store: {
        const LocalView view = makeLocalView(plan->storeSlotIndex,
                                             plan->storeOffset,
                                             plan->storeLayoutMode,
                                             plan->validM,
                                             plan->validN,
                                             plan->validK);
        MemRequestDesc req;
        req.portId = 0;
        req.addr = plan->storeSpmAddr;
        req.size = view.linearBytes;
        req.data = readLinearFromSlot(view);
        reqs.push_back(req);
        return;
      }
      case Mode::TensorLoop:
        if (plan->doStoreC) {
            MemRequestDesc req;
            req.portId = 0;
            req.addr = plan->tensorCAddr;
            req.size = plan->tensorCResult.size();
            req.data = plan->tensorCResult;
            reqs.push_back(req);
        }
        return;
      case Mode::Load:
      case Mode::Compute:
        return;
    }
}

void
MpuUnit::onMvoutResponse(ActiveExecution &exec, const MemTxnContext &txn,
                         PacketPtr pkt)
{
    (void)exec;
    (void)pkt;

    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Store: {
        SlotState &slot = slotByIndex(slotIndexForAddr(parsedCmd.localAddrC));
        slot.dirty = false;
        slot.busy = false;
        return;
      }
      case Mode::TensorLoop: {
        const IterationPlan &plan = iterationPlan(txn.iteration);
        if (plan.doStoreC) {
            SlotState &slot = slotByIndex(plan.tensorSlotC);
            slot.dirty = false;
            slot.busy = false;
        }
        return;
      }
      case Mode::Load:
      case Mode::Compute:
        return;
    }
}

uint64_t
MpuUnit::loadCmdCount() const
{
    return loadCount;
}

uint64_t
MpuUnit::computeCmdCount() const
{
    return computeCount;
}

uint64_t
MpuUnit::storeCmdCount() const
{
    return storeCount;
}

uint64_t
MpuUnit::tensorLoopCmdCount() const
{
    return tensorLoopCount;
}

uint64_t
MpuUnit::matmulCount() const
{
    return matmulCountValue;
}

uint64_t
MpuUnit::matmulAccCount() const
{
    return matmulAccCountValue;
}

uint64_t
MpuUnit::slotA0Valid() const
{
    return slots[SlotA0].valid;
}

uint64_t
MpuUnit::slotA1Valid() const
{
    return slots[SlotA1].valid;
}

uint64_t
MpuUnit::slotB0Valid() const
{
    return slots[SlotB0].valid;
}

uint64_t
MpuUnit::slotB1Valid() const
{
    return slots[SlotB1].valid;
}

uint64_t
MpuUnit::slotC0Valid() const
{
    return slots[SlotC0].valid;
}

uint64_t
MpuUnit::slotC1Valid() const
{
    return slots[SlotC1].valid;
}

uint64_t
MpuUnit::slotC0Dirty() const
{
    return slots[SlotC0].dirty;
}

uint64_t
MpuUnit::slotC1Dirty() const
{
    return slots[SlotC1].dirty;
}

uint64_t
MpuUnit::tensorLoopExpandedTiles() const
{
    return tensorLoopExpandedTilesCount;
}

uint64_t
MpuUnit::tensorLoopExpandedAccTiles() const
{
    return tensorLoopExpandedAccTilesCount;
}

uint64_t
MpuUnit::totalInternalLoads() const
{
    return totalInternalLoadsValue;
}

uint64_t
MpuUnit::totalInternalComputes() const
{
    return totalInternalComputesValue;
}

uint64_t
MpuUnit::totalInternalStores() const
{
    return totalInternalStoresValue;
}

uint64_t
MpuUnit::totalTiles() const
{
    return totalTilesValue;
}

uint64_t
MpuUnit::totalAccTiles() const
{
    return totalAccTilesValue;
}

uint64_t
MpuUnit::stallCyclesWaitingForSPM() const
{
    return stallCyclesWaitingForSPMValue;
}

uint64_t
MpuUnit::stallCyclesWaitingForSlot() const
{
    return stallCyclesWaitingForSlotValue;
}

uint64_t
MpuUnit::computedTotalLatency() const
{
    return computedTotalLatencyValue;
}

} // namespace gem5
