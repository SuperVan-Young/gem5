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

MpuUnit::MpuUnit(const MpuUnitParams &params)
    : SpecializedExecutionUnit(params),
      arrayRows(params.array_rows),
      arrayCols(params.array_cols),
      arrayKDepth(params.array_k_depth),
      loadBaseLatency(params.load_base_latency),
      storeBaseLatency(params.store_base_latency),
      arrayFillLatency(
          params.array_fill_latency ?
          *params.array_fill_latency :
          ((params.array_rows > 0 ? params.array_rows - 1 : 0) *
           params.clk_domain->clockPeriod())),
      arraySteadyPerK(
          params.array_steady_per_k ?
          *params.array_steady_per_k :
          params.clk_domain->clockPeriod()),
      arrayDrainLatency(
          params.array_drain_latency ?
          *params.array_drain_latency :
          ((params.array_cols > 0 ? params.array_cols - 1 : 0) *
           params.clk_domain->clockPeriod())),
      loadBandwidthBytesPerCycle(params.load_bandwidth_bytes_per_cycle),
      storeBandwidthBytesPerCycle(params.store_bandwidth_bytes_per_cycle),
      cReadBaseLatency(params.c_read_base_latency),
      cWriteBaseLatency(params.c_write_base_latency),
      localBankCount(params.local_bank_count),
      localBankGranularityBytes(params.local_bank_granularity_bytes),
      localBankServiceCycles(params.local_bank_service_cycles),
      parsedCmdValid(false),
      localBankReadyTicks(localBankCount, 0),
      memPortReadyTicks(params.num_mem_side_ports, 0),
      currentObservedMemWait(0),
      currentObservedSlotWait(0),
      currentObservedExecLatency(0),
      activeCmdStartTick(0),
      slotEpochCounter(1),
      reservedSlotMask(0),
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
      partialSumSpillCountValue(0),
      partialSumReloadCountValue(0),
      stallCyclesWaitingForSPMValue(0),
      stallCyclesWaitingForSlotValue(0),
      observedTotalLatencyValue(0),
      observedLoadServiceCyclesValue(0),
      observedStoreServiceCyclesValue(0),
      observedExecServiceCyclesValue(0),
      tensorLoopSlotUseMaskAValue(0),
      tensorLoopSlotUseMaskBValue(0),
      tensorLoopSlotUseMaskCValue(0)
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
        parsed.layoutMode = parsed.words[6] & LoadStoreLayoutMask;
        parsed.offsetA =
            (parsed.words[6] >> LoadStoreOffsetShift) & LoadStoreOffsetMask;
        break;
      case Mode::Compute:
        parsed.localAddrA = parsed.words[1];
        parsed.localAddrB = parsed.words[2];
        parsed.localAddrC = parsed.words[3];
        parsed.validM = parsed.words[4];
        parsed.validN = parsed.words[5];
        parsed.validK = parsed.words[6];
        parsed.subop = static_cast<ComputeSubop>(parsed.words[7]);
        parsed.dstLayoutMode = parsed.words[8];
        parsed.computeModeFlags = parsed.words[9];
        parsed.offsetA = parsed.words[10];
        parsed.offsetB = parsed.words[11];
        parsed.offsetC = parsed.words[12];
        break;
      case Mode::Store:
        parsed.localAddrC = parsed.words[1];
        parsed.spmAddrC = parsed.words[2];
        parsed.validM = parsed.words[3];
        parsed.validN = parsed.words[4];
        parsed.validK = parsed.words[5];
        parsed.layoutMode = parsed.words[6] & LoadStoreLayoutMask;
        parsed.offsetC =
            (parsed.words[6] >> LoadStoreOffsetShift) & LoadStoreOffsetMask;
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
        parsed.subop = static_cast<ComputeSubop>(parsed.words[10] &
                                                 LoopCtrl0SubopMask);
        parsed.tensorLayoutA =
            (parsed.words[10] >> LoopCtrl0LayoutAShift) & 0x3U;
        parsed.tensorLayoutB =
            (parsed.words[10] >> LoopCtrl0LayoutBShift) & 0x3U;
        parsed.tensorLayoutC =
            (parsed.words[10] >> LoopCtrl0LayoutCShift) & 0x3U;
        parsed.outerAxis = static_cast<Axis>(
            (parsed.words[10] >> LoopCtrl0AxisOuterShift) & 0x3U);
        parsed.innerAxis = static_cast<Axis>(
            (parsed.words[10] >> LoopCtrl0AxisInnerShift) & 0x3U);
        parsed.outerCount = parsed.words[11];
        parsed.innerCount = parsed.words[12];
        parsed.offsetA = parsed.words[13] & 0xffU;
        parsed.offsetB = (parsed.words[13] >> 8) & 0xffU;
        parsed.offsetC = (parsed.words[13] >> 16) & 0xffU;
        parsed.outerStepTiles = parsed.words[14] & 0xffU;
        parsed.innerStepTiles = (parsed.words[14] >> 8) & 0xffU;
        parsed.pingpongA = ((parsed.words[14] >> 16) & 0x1U) != 0;
        parsed.pingpongB = ((parsed.words[14] >> 17) & 0x1U) != 0;
        parsed.pingpongC = ((parsed.words[14] >> 18) & 0x1U) != 0;
        parsed.autoLoadCForAcc = ((parsed.words[14] >> 19) & 0x1U) != 0;
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
    pendingStoreReadyTicks.clear();
    pendingExecReadyTicks.clear();
    pendingComputeCommits.clear();
    currentObservedMemWait = 0;
    currentObservedSlotWait = 0;
    currentObservedExecLatency = 0;
    reservedSlotMask = 0;
    std::fill(localBankReadyTicks.begin(), localBankReadyTicks.end(), 0);
    std::fill(memPortReadyTicks.begin(), memPortReadyTicks.end(), 0);
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
    const uint32_t base_addr = local_addr;
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

void
MpuUnit::validateSlotWindow(const SlotState &slot, uint32_t offset_bytes,
                            uint32_t layout_mode, const ParsedCmd &cmd,
                            const char *label) const
{
    panic_if(slot.residentOffsetBytes != offset_bytes,
             "MpuUnit: %s slot offset mismatch slot=%u cmd=%u",
             label, slot.residentOffsetBytes, offset_bytes);
    panic_if(slot.residentLayoutMode != layout_mode,
             "MpuUnit: %s slot layout mismatch slot=%u cmd=%u",
             label, slot.residentLayoutMode, layout_mode);
    validateSlotShape(slot, cmd, label);
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
MpuUnit::updateSlotResidentWindow(SlotState &slot, uint32_t offset_bytes,
                                  uint32_t layout_mode, uint32_t valid_m,
                                  uint32_t valid_n, uint32_t valid_k,
                                  bool dirty)
{
    slot.valid = true;
    slot.dirty = dirty;
    slot.shapeM = valid_m;
    slot.shapeN = valid_n;
    slot.shapeK = valid_k;
    slot.residentOffsetBytes = offset_bytes;
    slot.residentLayoutMode = layout_mode;
    slot.epochId = slotEpochCounter++;
}

void
MpuUnit::reserveSlot(int slot_index, const char *label)
{
    SlotState &slot = slotByIndex(slot_index);
    panic_if(slot.busy,
             "MpuUnit: attempted to reserve busy slot %#x for %s",
             slot.localAddr, label);
    slot.busy = true;
    reservedSlotMask |= (1U << slot_index);
    DPRINTF(MpuUnit, "reserve slot=%#x label=%s mask=%#x\n",
            slot.localAddr, label, reservedSlotMask);
}

void
MpuUnit::releaseReservedSlots()
{
    const uint32_t releasing_mask = reservedSlotMask;
    for (size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
        if ((reservedSlotMask & (1U << slot_index)) == 0) {
            continue;
        }
        slots[slot_index].busy = false;
    }
    reservedSlotMask = 0;
    DPRINTF(MpuUnit, "release reserved slots mask=%#x\n", releasing_mask);
}

void
MpuUnit::reserveSlotsForCommand(const ParsedCmd &cmd)
{
    switch (static_cast<Mode>(cmd.mode)) {
      case Mode::Load:
        reserveSlot(slotIndexForAddr(cmd.localAddrA), "load");
        return;
      case Mode::Compute:
        reserveSlot(slotIndexForAddr(cmd.localAddrA), "compute-a");
        reserveSlot(slotIndexForAddr(cmd.localAddrB), "compute-b");
        reserveSlot(slotIndexForAddr(cmd.localAddrC), "compute-c");
        return;
      case Mode::Store:
        reserveSlot(slotIndexForAddr(cmd.localAddrC), "store");
        return;
      case Mode::TensorLoop: {
        int slot_a = slotIndexForAddr(cmd.localAddrA);
        int slot_b = slotIndexForAddr(cmd.localAddrB);
        int slot_c = slotIndexForAddr(cmd.localAddrC);
        reserveSlot(slot_a, "tensor-loop-a");
        reserveSlot(slot_b, "tensor-loop-b");
        reserveSlot(slot_c, "tensor-loop-c");
        if (cmd.pingpongA) {
            reserveSlot(alternateSlotIndex(slot_a), "tensor-loop-a-alt");
        }
        if (cmd.pingpongB) {
            reserveSlot(alternateSlotIndex(slot_b), "tensor-loop-b-alt");
        }
        if (cmd.pingpongC) {
            reserveSlot(alternateSlotIndex(slot_c), "tensor-loop-c-alt");
        }
        return;
      }
    }

    panic("MpuUnit: unreachable slot reservation");
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
    reserveSlotsForCommand(cmd);
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

std::vector<uint8_t>
MpuUnit::readLinearFromExternal(SlotKind kind, uint32_t layout_mode,
                                uint32_t valid_m, uint32_t valid_n,
                                uint32_t valid_k, const uint8_t *bytes,
                                size_t size) const
{
    LocalView view;
    view.kind = kind;
    view.layoutMode = layout_mode;
    view.validM = valid_m;
    view.validN = valid_n;
    view.validK = valid_k;
    switch (kind) {
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
    panic_if(size != view.physicalBytes,
             "MpuUnit: external read size=%zu expected=%zu", size,
             view.physicalBytes);

    std::vector<uint8_t> linear(view.linearBytes, 0);
    size_t linear_offset = 0;
    for (size_t row = 0; row < view.rows; ++row) {
        for (size_t col = 0; col < view.cols; ++col) {
            const size_t phys = row * view.rowStrideBytes +
                (col * view.elemBytes);
            std::memcpy(linear.data() + linear_offset, bytes + phys,
                        view.elemBytes);
            linear_offset += view.elemBytes;
        }
    }
    return linear;
}

std::vector<uint8_t>
MpuUnit::writeLinearToExternal(SlotKind kind, uint32_t layout_mode,
                               uint32_t valid_m, uint32_t valid_n,
                               uint32_t valid_k,
                               const std::vector<uint8_t> &linear_bytes) const
{
    LocalView view;
    view.kind = kind;
    view.layoutMode = layout_mode;
    view.validM = valid_m;
    view.validN = valid_n;
    view.validK = valid_k;
    switch (kind) {
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
    panic_if(linear_bytes.size() != view.linearBytes,
             "MpuUnit: external write size=%zu expected=%zu",
             linear_bytes.size(), view.linearBytes);

    std::vector<uint8_t> physical(view.physicalBytes, 0);
    size_t linear_offset = 0;
    for (size_t row = 0; row < view.rows; ++row) {
        for (size_t col = 0; col < view.cols; ++col) {
            const size_t phys = row * view.rowStrideBytes +
                (col * view.elemBytes);
            std::memcpy(physical.data() + phys,
                        linear_bytes.data() + linear_offset,
                        view.elemBytes);
            linear_offset += view.elemBytes;
        }
    }
    return physical;
}

Tick
MpuUnit::issueLocalAccess(const LocalView &view, Tick start_tick)
{
    Tick max_ready = start_tick;
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
                max_ready = std::max(max_ready, localBankReadyTicks[bank]);
                localBankReadyTicks[bank] = max_ready +
                    ticksForCycles(localBankServiceCycles);
            }
        }
    }

    const Tick stall = max_ready > start_tick ? max_ready - start_tick : 0;
    stallCyclesWaitingForSlotValue += stall;
    currentObservedSlotWait += stall;
    return max_ready;
}

Tick
MpuUnit::observedLoadExecLatency(const LocalView &view)
{
    Tick cursor = curTick() + loadBaseLatency;
    cursor = issueLocalAccess(view, cursor);
    return cursor - curTick();
}

Tick
MpuUnit::observedStoreExecLatency(const LocalView &view)
{
    Tick cursor = curTick() + storeBaseLatency;
    cursor = issueLocalAccess(view, cursor);
    return cursor - curTick();
}

Tick
MpuUnit::observedMatmulExecLatency(const LocalView &view_a,
                                   const LocalView &view_b,
                                   const LocalView &view_c,
                                   uint32_t valid_k,
                                   bool accumulate)
{
    Tick cursor = curTick();
    cursor = issueLocalAccess(view_a, cursor);
    cursor = issueLocalAccess(view_b, cursor);
    if (accumulate) {
        cursor += cReadBaseLatency;
        cursor = issueLocalAccess(view_c, cursor);
    }
    cursor += matmulLatency(valid_k);
    if (accumulate) {
        cursor += cWriteBaseLatency;
    }
    cursor = issueLocalAccess(view_c, cursor);
    return cursor - curTick();
}

uint64_t
MpuUnit::outputTileKey(uint32_t m_index, uint32_t n_index) const
{
    return (static_cast<uint64_t>(m_index) << 32) | n_index;
}

size_t
MpuUnit::externalTileBytes(SlotKind kind, uint32_t layout_mode,
                           uint32_t valid_m, uint32_t valid_n,
                           uint32_t valid_k) const
{
    LocalView view;
    view.kind = kind;
    view.layoutMode = layout_mode;
    switch (kind) {
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
        break;
    }
    view.rowStrideBytes = view.cols * view.elemBytes;
    if (layout_mode == static_cast<uint32_t>(LayoutMode::Skewed) &&
        view.rows > 1) {
        view.rowStrideBytes += view.elemBytes;
    }
    return view.rows == 0 ? 0 :
        ((view.rows - 1) * view.rowStrideBytes) +
        (view.cols * view.elemBytes);
}

void
MpuUnit::recordMemWait(PortID port_id, size_t bytes, uint32_t bytes_per_cycle)
{
    if (port_id < 0 || static_cast<size_t>(port_id) >= memPortReadyTicks.size()) {
        return;
    }
    const Tick ready = std::max(curTick(), memPortReadyTicks[port_id]);
    const Tick wait = ready > curTick() ? ready - curTick() : 0;
    stallCyclesWaitingForSPMValue += wait;
    currentObservedMemWait += wait;
    memPortReadyTicks[port_id] =
        ready + bandwidthLatency(bytes, bytes_per_cycle);
}

PortID
MpuUnit::selectTensorLoopPort(PendingLoadKind kind) const
{
    const size_t port_count = memSidePorts.size();
    if (port_count == 0) {
        return 0;
    }
    const size_t preferred =
        kind == PendingLoadKind::TensorLoadA ? 0 :
        kind == PendingLoadKind::TensorLoadB ? 1 : 2;
    return static_cast<PortID>(preferred % port_count);
}

void
MpuUnit::resetObservedState()
{
    currentObservedMemWait = 0;
    currentObservedSlotWait = 0;
    currentObservedExecLatency = 0;
    std::fill(localBankReadyTicks.begin(), localBankReadyTicks.end(), 0);
    std::fill(memPortReadyTicks.begin(), memPortReadyTicks.end(), 0);
}

void
MpuUnit::validateLoadCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    panic_if((cmd.localAddrA & 0xfU) != 0,
             "MpuUnit: load destination local address %#x must use slot base "
             "without low-bit offset", cmd.localAddrA);
    const int slot_index = slotIndexForAddr(cmd.localAddrA);
    panic_if(slot_index < 0,
             "MpuUnit: invalid load destination local address %#x",
             cmd.localAddrA);
    panic_if((cmd.words[6] & LoadStoreCtrlReservedMask) != 0,
             "MpuUnit: load_store_ctrl reserved bits must be zero");
    for (size_t i = 7; i < cmd.words.size(); ++i) {
        panic_if(cmd.words[i] != 0,
                 "MpuUnit: load reserved Word %zu must be zero, got %#x",
                 i, cmd.words[i]);
    }

    const SlotState &slot = slotByIndex(slot_index);
    loadViewForSlot(slot_index, cmd.layoutMode, cmd.offsetA,
                    cmd.validM, cmd.validN, cmd.validK);
    validateSpmAddress(cmd.spmAddrA,
                       externalTileBytes(slot.kind, cmd.layoutMode,
                                         cmd.validM, cmd.validN, cmd.validK),
                       "load source");
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
    panic_if((cmd.localAddrA & 0xfU) != 0 || (cmd.localAddrB & 0xfU) != 0 ||
             (cmd.localAddrC & 0xfU) != 0,
             "MpuUnit: compute local addresses must use slot base values");
    panic_if(cmd.computeModeFlags != 0,
             "MpuUnit: compute_mode_flags must be zero in v1, got %#x",
             cmd.computeModeFlags);
    panic_if(cmd.dstLayoutMode > static_cast<uint32_t>(LayoutMode::Skewed),
             "MpuUnit: unsupported compute dst_layout_mode=%u",
             cmd.dstLayoutMode);
    for (size_t i = 13; i < cmd.words.size(); ++i) {
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
    validateSlotWindow(slot_a, cmd.offsetA, slot_a.residentLayoutMode, cmd, "A");
    validateSlotWindow(slot_b, cmd.offsetB, slot_b.residentLayoutMode, cmd, "B");
    makeLocalView(slotIndexForAddr(cmd.localAddrA), cmd.offsetA,
                  slot_a.residentLayoutMode, cmd.validM, cmd.validN,
                  cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrB), cmd.offsetB,
                  slot_b.residentLayoutMode, cmd.validM, cmd.validN,
                  cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrC), cmd.offsetC,
                  cmd.dstLayoutMode, cmd.validM, cmd.validN, cmd.validK);
    if (cmd.subop == ComputeSubop::MatmulAcc) {
        panic_if(!slot_c.valid, "MpuUnit: MATMUL_ACC requires valid C slot");
        validateSlotWindow(slot_c, cmd.offsetC, cmd.dstLayoutMode, cmd, "C");
    }
}

void
MpuUnit::validateStoreCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    validateSlotAddress(cmd.localAddrC, SlotKind::C, "store source");
    panic_if((cmd.localAddrC & 0xfU) != 0,
             "MpuUnit: store source local address %#x must use slot base "
             "without low-bit offset", cmd.localAddrC);
    panic_if((cmd.words[6] & LoadStoreCtrlReservedMask) != 0,
             "MpuUnit: load_store_ctrl reserved bits must be zero");
    for (size_t i = 7; i < cmd.words.size(); ++i) {
        panic_if(cmd.words[i] != 0,
                 "MpuUnit: store reserved Word %zu must be zero, got %#x",
                 i, cmd.words[i]);
    }

    const SlotState &slot = slotByIndex(slotIndexForAddr(cmd.localAddrC));
    panic_if(!slot.valid, "MpuUnit: store source C slot is invalid");
    panic_if(slot.busy, "MpuUnit: store source C slot is busy");
    validateSlotWindow(slot, cmd.offsetC, slot.residentLayoutMode, cmd,
                       "store source");
    makeLocalView(slotIndexForAddr(cmd.localAddrC),
                  cmd.offsetC,
                  slot.residentLayoutMode, cmd.validM,
                  cmd.validN, cmd.validK);
    validateSpmAddress(cmd.spmAddrC,
                       externalTileBytes(SlotKind::C, cmd.layoutMode,
                                         cmd.validM, cmd.validN, cmd.validK),
                       "store destination");
}

void
MpuUnit::validateTensorLoopCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    validateSlotAddress(cmd.localAddrA, SlotKind::A, "tensor_loop base A");
    validateSlotAddress(cmd.localAddrB, SlotKind::B, "tensor_loop base B");
    validateSlotAddress(cmd.localAddrC, SlotKind::C, "tensor_loop base C");
    panic_if((cmd.localAddrA & 0xfU) != 0 || (cmd.localAddrB & 0xfU) != 0 ||
             (cmd.localAddrC & 0xfU) != 0,
             "MpuUnit: tensor_loop base local addresses must use slot base values");
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
    panic_if((cmd.words[10] & LoopCtrl0ReservedMask) != 0,
             "MpuUnit: tensor_loop loop_ctrl0 reserved bits must be zero");
    panic_if((cmd.words[13] & 0xff000000U) != 0,
             "MpuUnit: tensor_loop offset_pack reserved bits must be zero");
    panic_if((cmd.words[14] & StepCfgReservedMask) != 0,
             "MpuUnit: tensor_loop step_cfg reserved bits must be zero");
    panic_if(cmd.words[15] != 0,
             "MpuUnit: tensor_loop reserved Word 15 must be zero");

    makeLocalView(slotIndexForAddr(cmd.localAddrA), cmd.offsetA,
                  cmd.tensorLayoutA, cmd.validM, cmd.validN, cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrB), cmd.offsetB,
                  cmd.tensorLayoutB, cmd.validM, cmd.validN, cmd.validK);
    makeLocalView(slotIndexForAddr(cmd.localAddrC), cmd.offsetC,
                  cmd.tensorLayoutC, cmd.validM, cmd.validN, cmd.validK);
    if (cmd.pingpongA) {
        makeLocalView(alternateSlotIndex(slotIndexForAddr(cmd.localAddrA)),
                      cmd.offsetA, cmd.tensorLayoutA,
                      cmd.validM, cmd.validN, cmd.validK);
    }
    if (cmd.pingpongB) {
        makeLocalView(alternateSlotIndex(slotIndexForAddr(cmd.localAddrB)),
                      cmd.offsetB, cmd.tensorLayoutB,
                      cmd.validM, cmd.validN, cmd.validK);
    }
    if (cmd.pingpongC) {
        makeLocalView(alternateSlotIndex(slotIndexForAddr(cmd.localAddrC)),
                      cmd.offsetC, cmd.tensorLayoutC,
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
        return externalTileBytes(SlotKind::A, parsedCmd.tensorLayoutA,
                                 valid_m, valid_n, valid_k);
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
        return externalTileBytes(SlotKind::B, parsedCmd.tensorLayoutB,
                                 valid_m, valid_n, valid_k);
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
        return externalTileBytes(SlotKind::C, parsedCmd.tensorLayoutC,
                                 valid_m, valid_n, valid_k);
      case Axis::K:
        return 0;
    }

    panic("MpuUnit: unreachable C axis delta");
}

void
MpuUnit::buildTensorLoopPlans()
{
    const bool includes_k =
        parsedCmd.outerAxis == Axis::K || parsedCmd.innerAxis == Axis::K;
    const bool k_outer = includes_k && parsedCmd.outerAxis == Axis::K;
    const uint32_t k_total = !includes_k ? 1U :
        (k_outer ? parsedCmd.outerCount : parsedCmd.innerCount);
    const uint32_t n_extent =
        parsedCmd.outerAxis == Axis::N ?
            ((parsedCmd.outerCount - 1U) * parsedCmd.outerStepTiles) + 1U :
        parsedCmd.innerAxis == Axis::N ?
            ((parsedCmd.innerCount - 1U) * parsedCmd.innerStepTiles) + 1U :
            1U;
    const uint32_t k_extent =
        parsedCmd.outerAxis == Axis::K ?
            ((parsedCmd.outerCount - 1U) * parsedCmd.outerStepTiles) + 1U :
        parsedCmd.innerAxis == Axis::K ?
            ((parsedCmd.innerCount - 1U) * parsedCmd.innerStepTiles) + 1U :
            1U;
    const Addr a_tile_bytes = externalTileBytes(SlotKind::A,
                                                parsedCmd.tensorLayoutA,
                                                parsedCmd.validM,
                                                parsedCmd.validN,
                                                parsedCmd.validK);
    const Addr b_tile_bytes = externalTileBytes(SlotKind::B,
                                                parsedCmd.tensorLayoutB,
                                                parsedCmd.validM,
                                                parsedCmd.validN,
                                                parsedCmd.validK);
    const Addr c_tile_bytes = externalTileBytes(SlotKind::C,
                                                parsedCmd.tensorLayoutC,
                                                parsedCmd.validM,
                                                parsedCmd.validN,
                                                parsedCmd.validK);
    std::unordered_map<uint64_t, int> output_to_c_slot;
    uint64_t output_tile_ordinal = 0;

    for (uint32_t outer = 0; outer < parsedCmd.outerCount; ++outer) {
        for (uint32_t inner = 0; inner < parsedCmd.innerCount; ++inner) {
            uint32_t m_index = 0;
            uint32_t n_index = 0;
            uint32_t k_index = 0;
            const uint32_t outer_value = outer * parsedCmd.outerStepTiles;
            const uint32_t inner_value = inner * parsedCmd.innerStepTiles;

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
            apply_axis(parsedCmd.outerAxis, outer_value);
            apply_axis(parsedCmd.innerAxis, inner_value);

            const Addr inner_a = parsedCmd.spmAddrA +
                (((static_cast<Addr>(m_index) * k_extent) + k_index) *
                 a_tile_bytes);
            const Addr inner_b = parsedCmd.spmAddrB +
                (((static_cast<Addr>(k_index) * n_extent) + n_index) *
                 b_tile_bytes);
            const Addr inner_c = parsedCmd.spmAddrC +
                (((static_cast<Addr>(m_index) * n_extent) + n_index) *
                 c_tile_bytes);

            validateSpmAddress(inner_a,
                               a_tile_bytes,
                               "tensor_loop A tile");
            validateSpmAddress(inner_b,
                               b_tile_bytes,
                               "tensor_loop B tile");
            validateSpmAddress(inner_c,
                               c_tile_bytes,
                               "tensor_loop C tile");

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
            const uint32_t k_position = !includes_k ? 0U :
                (k_outer ? outer : inner);
            const bool last_k = !includes_k || (k_position + 1U == k_total);
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
            plan.tensorOffsetA = parsedCmd.offsetA;
            plan.tensorOffsetB = parsedCmd.offsetB;
            plan.tensorOffsetC = parsedCmd.offsetC;
            plan.tensorLayoutA = parsedCmd.tensorLayoutA;
            plan.tensorLayoutB = parsedCmd.tensorLayoutB;
            plan.tensorLayoutC = parsedCmd.tensorLayoutC;
            plan.tensorAAddr = inner_a;
            plan.tensorBAddr = inner_b;
            plan.tensorCAddr = inner_c;
            plan.firstK = first_k;
            plan.lastK = last_k;
            plan.kExpanded = includes_k;
            plan.kOuter = k_outer;
            plan.outputTileKey = out_key;
            plan.doLoadA = true;
            plan.doLoadB = true;
            if (!includes_k) {
                plan.subop = parsedCmd.subop;
                plan.doLoadCOld = parsedCmd.subop == ComputeSubop::MatmulAcc &&
                    parsedCmd.autoLoadCForAcc;
                plan.doStoreC = true;
                plan.finalOutputStore = true;
            } else if (k_outer) {
                plan.subop = first_k ? ComputeSubop::Matmul
                                     : ComputeSubop::MatmulAcc;
                plan.doLoadCOld =
                    (!first_k) ||
                    (parsedCmd.subop == ComputeSubop::MatmulAcc &&
                     parsedCmd.autoLoadCForAcc && first_k);
                plan.doStoreC = true;
                plan.kOuterSpill = !last_k;
                plan.finalOutputStore = last_k;
            } else {
                plan.subop = first_k ? ComputeSubop::Matmul
                                     : ComputeSubop::MatmulAcc;
                plan.doLoadCOld =
                    parsedCmd.subop == ComputeSubop::MatmulAcc &&
                    parsedCmd.autoLoadCForAcc && first_k;
                plan.doStoreC = last_k;
                plan.finalOutputStore = last_k;
            }
            iterationPlans.push_back(std::move(plan));
        }
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
        plan.loadOffset = parsedCmd.offsetA;
        plan.loadLayoutMode = parsedCmd.layoutMode;
        plan.loadSpmAddr = parsedCmd.spmAddrA;
        iterationPlans.push_back(std::move(plan));
        exec.readMask = 0U;
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
        plan.computeOffsetA = parsedCmd.offsetA;
        plan.computeOffsetB = parsedCmd.offsetB;
        plan.computeOffsetC = parsedCmd.offsetC;
        plan.subop = parsedCmd.subop;
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
        plan.storeOffset = parsedCmd.offsetC;
        plan.storeLayoutMode = parsedCmd.layoutMode;
        plan.storeSpmAddr = parsedCmd.spmAddrC;
        iterationPlans.push_back(std::move(plan));
        exec.readMask = 0U;
        exec.writeMask = 0U;
        exec.repetition = 1;
        break;
      }
      case Mode::TensorLoop:
        buildTensorLoopPlans();
        exec.readMask = 0U;
        exec.writeMask = 0U;
        exec.repetition = iterationPlans.size();
        break;
    }
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
    activeCmdStartTick = curTick();
    parsedCmd = parseCommand(exec.cmd);
    parsedCmdValid = true;
    validateParsedCommand(parsedCmd);

    buildIterationPlans(exec);

    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Load:
        loadCount++;
        markBusyForFineCommand(parsedCmd);
        totalInternalLoadsValue++;
        break;
      case Mode::Compute:
        computeCount++;
        markBusyForFineCommand(parsedCmd);
        totalInternalComputesValue++;
        totalTilesValue++;
        if (parsedCmd.subop == ComputeSubop::MatmulAcc) {
            totalAccTilesValue++;
        }
        break;
      case Mode::Store:
        storeCount++;
        markBusyForFineCommand(parsedCmd);
        totalInternalStoresValue++;
        break;
      case Mode::TensorLoop:
        tensorLoopCount++;
        reserveSlotsForCommand(parsedCmd);
        tensorLoopExpandedTilesCount += iterationPlans.size();
        for (const auto &plan : iterationPlans) {
            tensorLoopSlotUseMaskAValue |= (1U << plan.tensorSlotA);
            tensorLoopSlotUseMaskBValue |= (1U << plan.tensorSlotB);
            tensorLoopSlotUseMaskCValue |= (1U << plan.tensorSlotC);
            tensorLoopExpandedAccTilesCount +=
                plan.subop == ComputeSubop::MatmulAcc ? 1U : 0U;
            totalInternalLoadsValue +=
                (plan.doLoadA ? 1U : 0U) + (plan.doLoadB ? 1U : 0U) +
                (plan.doLoadCOld ? 1U : 0U);
            totalInternalComputesValue++;
            totalInternalStoresValue += plan.doStoreC ? 1U : 0U;
            totalTilesValue++;
            totalAccTilesValue +=
                plan.subop == ComputeSubop::MatmulAcc ? 1U : 0U;
            if (plan.kOuterSpill) {
                partialSumSpillCountValue++;
            }
            if (plan.kOuter && !plan.firstK) {
                partialSumReloadCountValue++;
            }
        }
        break;
    }
}

void
MpuUnit::epilogue(ActiveExecution &exec)
{
    if (exec.iteration + 1 == exec.repetition) {
        observedTotalLatencyValue += curTick() - activeCmdStartTick;
        panic_if(reservedSlotMask != 0 && slotBusyMask() != reservedSlotMask,
                 "MpuUnit: reserved slot mask mismatch before release "
                 "reserved=%#x busy=%#llx",
                 reservedSlotMask,
                 static_cast<unsigned long long>(slotBusyMask()));
        releaseReservedSlots();
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
        MemRequestDesc req;
        req.portId = 0;
        req.addr = plan->loadSpmAddr;
        req.size = externalTileBytes(slotByIndex(plan->loadSlotIndex).kind,
                                     plan->loadLayoutMode, plan->validM,
                                     plan->validN, plan->validK);
        reqs.push_back(req);
        recordMemWait(req.portId, req.size, loadBandwidthBytesPerCycle);
        pendingLoadTxns.emplace(token, PendingLoadTxn{
            exec.iteration, PendingLoadKind::SlotLoad, plan->loadSlotIndex,
            curTick()});
        break;
      }
      case Mode::TensorLoop:
        if (plan->doLoadA) {
            MemRequestDesc req;
            req.portId = selectTensorLoopPort(PendingLoadKind::TensorLoadA);
            req.addr = plan->tensorAAddr;
            req.size = externalTileBytes(SlotKind::A, plan->tensorLayoutA,
                                         plan->validM, plan->validN,
                                         plan->validK);
            reqs.push_back(req);
            recordMemWait(req.portId, req.size, loadBandwidthBytesPerCycle);
            pendingLoadTxns.emplace(token++, PendingLoadTxn{
                exec.iteration, PendingLoadKind::TensorLoadA,
                plan->tensorSlotA, curTick()});
        }
        if (plan->doLoadB) {
            MemRequestDesc req;
            req.portId = selectTensorLoopPort(PendingLoadKind::TensorLoadB);
            req.addr = plan->tensorBAddr;
            req.size = externalTileBytes(SlotKind::B, plan->tensorLayoutB,
                                         plan->validM, plan->validN,
                                         plan->validK);
            reqs.push_back(req);
            recordMemWait(req.portId, req.size, loadBandwidthBytesPerCycle);
            pendingLoadTxns.emplace(token++, PendingLoadTxn{
                exec.iteration, PendingLoadKind::TensorLoadB,
                plan->tensorSlotB, curTick()});
        }
        if (plan->doLoadCOld) {
            MemRequestDesc req;
            req.portId = selectTensorLoopPort(PendingLoadKind::TensorLoadCOld);
            req.addr = plan->tensorCAddr;
            req.size = externalTileBytes(SlotKind::C, plan->tensorLayoutC,
                                         plan->validM, plan->validN,
                                         plan->validK);
            reqs.push_back(req);
            recordMemWait(req.portId, req.size, loadBandwidthBytesPerCycle);
            pendingLoadTxns.emplace(token, PendingLoadTxn{
                exec.iteration, PendingLoadKind::TensorLoadCOld,
                plan->tensorSlotC, curTick()});
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
        std::vector<uint8_t> linear = readLinearFromExternal(
            slotByIndex(plan.loadSlotIndex).kind, plan.loadLayoutMode,
            plan.validM, plan.validN, plan.validK, data, pkt->getSize());
        SlotState &slot = slotByIndex(plan.loadSlotIndex);
        writeLinearToSlot(view, linear);
        updateSlotResidentWindow(slot, plan.loadOffset, plan.loadLayoutMode,
                                 plan.validM, plan.validN, plan.validK, false);
        break;
      }
      case PendingLoadKind::TensorLoadA: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        const LocalView view = makeLocalView(plan.tensorSlotA, plan.tensorOffsetA,
                                             plan.tensorLayoutA, plan.validM,
                                             plan.validN, plan.validK);
        std::vector<uint8_t> linear = readLinearFromExternal(
            SlotKind::A, plan.tensorLayoutA, plan.validM, plan.validN,
            plan.validK, data, pkt->getSize());
        SlotState &slot = slotByIndex(plan.tensorSlotA);
        writeLinearToSlot(view, linear);
        updateSlotResidentWindow(slot, plan.tensorOffsetA, plan.tensorLayoutA,
                                 plan.validM, plan.validN, plan.validK, false);
        break;
      }
      case PendingLoadKind::TensorLoadB: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        const LocalView view = makeLocalView(plan.tensorSlotB, plan.tensorOffsetB,
                                             plan.tensorLayoutB, plan.validM,
                                             plan.validN, plan.validK);
        std::vector<uint8_t> linear = readLinearFromExternal(
            SlotKind::B, plan.tensorLayoutB, plan.validM, plan.validN,
            plan.validK, data, pkt->getSize());
        SlotState &slot = slotByIndex(plan.tensorSlotB);
        writeLinearToSlot(view, linear);
        updateSlotResidentWindow(slot, plan.tensorOffsetB, plan.tensorLayoutB,
                                 plan.validM, plan.validN, plan.validK, false);
        break;
      }
      case PendingLoadKind::TensorLoadCOld: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        const LocalView view = makeLocalView(plan.tensorSlotC, plan.tensorOffsetC,
                                             plan.tensorLayoutC, plan.validM,
                                             plan.validN, plan.validK);
        std::vector<uint8_t> linear = readLinearFromExternal(
            SlotKind::C, plan.tensorLayoutC, plan.validM, plan.validN,
            plan.validK, data, pkt->getSize());
        SlotState &slot = slotByIndex(plan.tensorSlotC);
        writeLinearToSlot(view, linear);
        updateSlotResidentWindow(slot, plan.tensorOffsetC, plan.tensorLayoutC,
                                 plan.validM, plan.validN, plan.validK, false);
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
        return observedLoadExecLatency(makeLocalView(plan.loadSlotIndex,
                                                     plan.loadOffset,
                                                     plan.loadLayoutMode,
                                                     plan.validM, plan.validN,
                                                     plan.validK));
      case Mode::Compute: {
        const SlotState &slot_a = slotByIndex(plan.computeSlotA);
        const SlotState &slot_b = slotByIndex(plan.computeSlotB);
        const LocalView view_a = makeLocalView(plan.computeSlotA,
                                               plan.computeOffsetA,
                                               slot_a.residentLayoutMode,
                                               plan.validM, plan.validN,
                                               plan.validK);
        const LocalView view_b = makeLocalView(plan.computeSlotB,
                                               plan.computeOffsetB,
                                               slot_b.residentLayoutMode,
                                               plan.validM, plan.validN,
                                               plan.validK);
        const LocalView view_c = makeLocalView(plan.computeSlotC,
                                               plan.computeOffsetC,
                                               parsedCmd.dstLayoutMode,
                                               plan.validM, plan.validN,
                                               plan.validK);
        const std::vector<uint8_t> a_bytes = readLinearFromSlot(view_a);
        const std::vector<uint8_t> b_bytes = readLinearFromSlot(view_b);
        std::vector<uint8_t> c_bytes =
            plan.subop == ComputeSubop::MatmulAcc ?
            readLinearFromSlot(view_c) : std::vector<uint8_t>();
        runMatmul(a_bytes, b_bytes, c_bytes, plan.validM, plan.validN,
                  plan.validK, plan.subop == ComputeSubop::MatmulAcc);
        pendingComputeCommits[exec.iteration] = PendingComputeCommit{
            plan.computeSlotC,
            plan.computeOffsetC,
            parsedCmd.dstLayoutMode,
            plan.validM,
            plan.validN,
            plan.validK,
            true,
            std::move(c_bytes),
        };
        pendingExecReadyTicks[exec.iteration] = curTick();
        const Tick latency = observedMatmulExecLatency(
            view_a, view_b, view_c, plan.validK,
            plan.subop == ComputeSubop::MatmulAcc);
        if (plan.subop == ComputeSubop::Matmul) {
            matmulCountValue++;
            return latency;
        }

        matmulAccCountValue++;
        return latency;
      }
      case Mode::Store:
        return observedStoreExecLatency(makeLocalView(plan.storeSlotIndex,
                                                      plan.storeOffset,
                                                      slotByIndex(plan.storeSlotIndex).residentLayoutMode,
                                                      plan.validM, plan.validN,
                                                      plan.validK));
      case Mode::TensorLoop: {
        const LocalView view_c = makeLocalView(plan.tensorSlotC,
                                               plan.tensorOffsetC,
                                               plan.tensorLayoutC,
                                               plan.validM, plan.validN,
                                               plan.validK);
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
        const std::vector<uint8_t> a_bytes = readLinearFromSlot(view_a);
        const std::vector<uint8_t> b_bytes = readLinearFromSlot(view_b);
        std::vector<uint8_t> c_bytes =
            plan.subop == ComputeSubop::MatmulAcc ?
            readLinearFromSlot(view_c) : std::vector<uint8_t>();
        runMatmul(a_bytes, b_bytes, c_bytes, plan.validM, plan.validN,
                  plan.validK, plan.subop == ComputeSubop::MatmulAcc);
        pendingComputeCommits[exec.iteration] = PendingComputeCommit{
            plan.tensorSlotC,
            plan.tensorOffsetC,
            plan.tensorLayoutC,
            plan.validM,
            plan.validN,
            plan.validK,
            true,
            std::move(c_bytes),
        };
        pendingExecReadyTicks[exec.iteration] = curTick();
        if (plan.subop == ComputeSubop::MatmulAcc) {
            matmulAccCountValue++;
        } else {
            matmulCountValue++;
        }
        return observedMatmulExecLatency(view_a, view_b, view_c, plan.validK,
                                         plan.subop == ComputeSubop::MatmulAcc);
      }
    }

    panic("MpuUnit: unreachable execute mode");
}

void
MpuUnit::commitPendingCompute(uint64_t iteration)
{
    auto it = pendingComputeCommits.find(iteration);
    if (it == pendingComputeCommits.end()) {
        return;
    }

    PendingComputeCommit &commit = it->second;
    SlotState &slot = slotByIndex(commit.slotIndex);
    const LocalView view = makeLocalView(commit.slotIndex, commit.offsetBytes,
                                         commit.layoutMode, commit.validM,
                                         commit.validN, commit.validK);
    writeLinearToSlot(view, commit.linearBytes);
    updateSlotResidentWindow(slot, commit.offsetBytes, commit.layoutMode,
                             commit.validM, commit.validN, commit.validK,
                             commit.dirty);
    pendingComputeCommits.erase(it);
}

void
MpuUnit::onMicroOpComplete(ActiveExecution &exec,
                           const MicroOpContext &ctx,
                           PacketPtr pkt)
{
    (void)exec;
    (void)pkt;

    switch (ctx.kind) {
      case MicroOpContext::Kind::Load: {
        auto it = pendingLoadTxns.find(ctx.token);
        if (it != pendingLoadTxns.end()) {
            observedLoadServiceCyclesValue += curTick() - it->second.readyTick;
            pendingLoadTxns.erase(it);
        }
        return;
      }
      case MicroOpContext::Kind::Store: {
        auto it = pendingStoreReadyTicks.find(ctx.token);
        if (it != pendingStoreReadyTicks.end()) {
            observedStoreServiceCyclesValue += curTick() - it->second;
            pendingStoreReadyTicks.erase(it);
        }
        return;
      }
      case MicroOpContext::Kind::Exec: {
        auto it = pendingExecReadyTicks.find(ctx.iteration);
        if (it != pendingExecReadyTicks.end()) {
            observedExecServiceCyclesValue += curTick() - it->second;
            pendingExecReadyTicks.erase(it);
        }
        commitPendingCompute(ctx.iteration);
        return;
      }
      case MicroOpContext::Kind::SyncWrite:
        return;
    }

    panic("MpuUnit: unreachable micro-op completion kind");
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
        const SlotState &slot = slotByIndex(plan->storeSlotIndex);
        const LocalView view = makeLocalView(plan->storeSlotIndex,
                                             plan->storeOffset,
                                             slot.residentLayoutMode,
                                             plan->validM,
                                             plan->validN,
                                             plan->validK);
        MemRequestDesc req;
        req.portId = 0;
        req.addr = plan->storeSpmAddr;
        req.data = writeLinearToExternal(SlotKind::C, plan->storeLayoutMode,
                                         plan->validM, plan->validN,
                                         plan->validK,
                                         readLinearFromSlot(view));
        req.size = req.data.size();
        reqs.push_back(req);
        recordMemWait(req.portId, req.size, storeBandwidthBytesPerCycle);
        pendingStoreReadyTicks.emplace(nextMemTxnToken, curTick());
        return;
      }
      case Mode::TensorLoop:
        if (plan->doStoreC) {
            const LocalView view = makeLocalView(plan->tensorSlotC,
                                                 plan->tensorOffsetC,
                                                 plan->tensorLayoutC,
                                                 plan->validM,
                                                 plan->validN,
                                                 plan->validK);
            MemRequestDesc req;
            req.portId = selectTensorLoopPort(PendingLoadKind::TensorLoadCOld);
            req.addr = plan->tensorCAddr;
            req.data = writeLinearToExternal(SlotKind::C, plan->tensorLayoutC,
                                             plan->validM, plan->validN,
                                             plan->validK,
                                             readLinearFromSlot(view));
            req.size = req.data.size();
            reqs.push_back(req);
            recordMemWait(req.portId, req.size, storeBandwidthBytesPerCycle);
            pendingStoreReadyTicks.emplace(nextMemTxnToken, curTick());
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
        return;
      }
      case Mode::TensorLoop: {
        const IterationPlan &plan = iterationPlan(txn.iteration);
        if (plan.doStoreC) {
            SlotState &slot = slotByIndex(plan.tensorSlotC);
            slot.dirty = false;
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
MpuUnit::observedTotalLatency() const
{
    return observedTotalLatencyValue;
}

uint64_t
MpuUnit::observedLoadServiceCycles() const
{
    return observedLoadServiceCyclesValue;
}

uint64_t
MpuUnit::observedStoreServiceCycles() const
{
    return observedStoreServiceCyclesValue;
}

uint64_t
MpuUnit::observedExecServiceCycles() const
{
    return observedExecServiceCyclesValue;
}

uint64_t
MpuUnit::partialSumSpillCount() const
{
    return partialSumSpillCountValue;
}

uint64_t
MpuUnit::partialSumReloadCount() const
{
    return partialSumReloadCountValue;
}

uint64_t
MpuUnit::slotBusyMask() const
{
    uint64_t mask = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].busy) {
            mask |= (1ULL << i);
        }
    }
    return mask;
}

uint64_t
MpuUnit::tensorLoopSlotUseMaskA() const
{
    return tensorLoopSlotUseMaskAValue;
}

uint64_t
MpuUnit::tensorLoopSlotUseMaskB() const
{
    return tensorLoopSlotUseMaskBValue;
}

uint64_t
MpuUnit::tensorLoopSlotUseMaskC() const
{
    return tensorLoopSlotUseMaskCValue;
}

} // namespace gem5
