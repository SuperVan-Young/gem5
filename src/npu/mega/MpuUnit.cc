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
#include <utility>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/MpuUnit.hh"

namespace gem5
{

namespace
{

} // namespace

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
      parsedCmdValid(false),
      loadCount(0),
      computeCount(0),
      storeCount(0),
      tensorLoopCount(0),
      matmulCountValue(0),
      matmulAccCountValue(0),
      tensorLoopExpandedTilesCount(0)
{
    fatal_if(macroCmdBytes != 64, "%s: MpuUnit requires 64-byte commands",
             name());
    fatal_if(arrayRows == 0 || arrayCols == 0 || arrayKDepth == 0,
             "%s: array dimensions must be non-zero", name());
    fatal_if(memSidePorts.empty(), "%s: MpuUnit requires at least one "
             "mem_side port", name());

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
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].localAddr == local_addr) {
            return static_cast<int>(i);
        }
    }

    return -1;
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
                              uint32_t valid_n,
                              uint32_t valid_k) const
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
    const SlotState &slot = slotByIndex(slot_index);
    switch (slot.kind) {
      case SlotKind::A:
        return aTileBytes(arrayRows, arrayKDepth);
      case SlotKind::B:
        return bTileBytes(arrayKDepth, arrayCols);
      case SlotKind::C:
        return cTileBytes(arrayRows, arrayCols);
    }

    panic("MpuUnit: unreachable slot capacity query");
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
    return loadBaseLatency + matmulLatency(valid_k) + storeBaseLatency;
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
                const int8_t a =
                    static_cast<int8_t>(a_bytes[static_cast<size_t>(m) *
                                                valid_k + k]);
                const int8_t b =
                    static_cast<int8_t>(b_bytes[static_cast<size_t>(k) *
                                                valid_n + n]);
                value += static_cast<int32_t>(a) * static_cast<int32_t>(b);
            }
            writeInt32(c_bytes, static_cast<size_t>(m) * valid_n + n, value);
        }
    }
}

void
MpuUnit::validateLoadCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    const int slot_index = slotIndexForAddr(cmd.localAddrA);
    panic_if(slot_index < 0,
             "MpuUnit: invalid load destination local address %#x",
             cmd.localAddrA);
    panic_if((cmd.localAddrA & 0xf) != 0,
             "MpuUnit: load destination local offset must be zero");
    panic_if(cmd.layoutMode != LayoutModeNormal,
             "MpuUnit: load layout_mode=%u is unsupported", cmd.layoutMode);
    for (size_t i = 7; i < cmd.words.size(); ++i) {
        panic_if(cmd.words[i] != 0,
                 "MpuUnit: load reserved Word %zu must be zero, got %#x",
                 i, cmd.words[i]);
    }

    const SlotState &slot = slotByIndex(slot_index);
    const size_t bytes = tileBytesForSlotKind(slot.kind, cmd.validM, cmd.validN,
                                              cmd.validK);
    panic_if(bytes > slot.bytes.size(),
             "MpuUnit: load payload %zu exceeds slot capacity %zu",
             bytes, slot.bytes.size());
    validateSpmAddress(cmd.spmAddrA, bytes, "load source");
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
    panic_if((cmd.localAddrA & 0xf) != 0 || (cmd.localAddrB & 0xf) != 0 ||
             (cmd.localAddrC & 0xf) != 0,
             "MpuUnit: compute local offsets must all be zero");
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
    panic_if((cmd.localAddrC & 0xf) != 0,
             "MpuUnit: store source local offset must be zero");
    panic_if(cmd.layoutMode != LayoutModeNormal,
             "MpuUnit: store layout_mode=%u is unsupported", cmd.layoutMode);
    for (size_t i = 7; i < cmd.words.size(); ++i) {
        panic_if(cmd.words[i] != 0,
                 "MpuUnit: store reserved Word %zu must be zero, got %#x",
                 i, cmd.words[i]);
    }

    const SlotState &slot = slotByIndex(slotIndexForAddr(cmd.localAddrC));
    panic_if(!slot.valid, "MpuUnit: store source C slot is invalid");
    panic_if(slot.busy, "MpuUnit: store source C slot is busy");
    validateSlotShape(slot, cmd, "store source");
    validateSpmAddress(cmd.spmAddrC, cTileBytes(cmd.validM, cmd.validN),
                       "store destination");
}

void
MpuUnit::validateTensorLoopCommand(const ParsedCmd &cmd) const
{
    validateValidShape(cmd);
    validateSlotAddress(cmd.localAddrA, SlotKind::A, "tensor_loop base A");
    validateSlotAddress(cmd.localAddrB, SlotKind::B, "tensor_loop base B");
    validateSlotAddress(cmd.localAddrC, SlotKind::C, "tensor_loop base C");
    panic_if((cmd.localAddrA & 0xf) != 0 || (cmd.localAddrB & 0xf) != 0 ||
             (cmd.localAddrC & 0xf) != 0,
             "MpuUnit: tensor_loop local offsets must all be zero");
    panic_if(!spmContains(cmd.spmAddrA, 1) || !spmContains(cmd.spmAddrB, 1) ||
             !spmContains(cmd.spmAddrC, 1),
             "MpuUnit: tensor_loop base SPM address is invalid");
    panic_if(cmd.subop != ComputeSubop::Matmul,
             "MpuUnit: Phase 3 tensor_loop only supports MATMUL");
    panic_if(cmd.outerCount == 0 || cmd.innerCount == 0,
             "MpuUnit: tensor_loop counts must be greater than zero");
    panic_if(static_cast<uint32_t>(cmd.outerAxis) >
                 static_cast<uint32_t>(Axis::K) ||
             static_cast<uint32_t>(cmd.innerAxis) >
                 static_cast<uint32_t>(Axis::K),
             "MpuUnit: tensor_loop axis must be one of M/N/K");
    panic_if(cmd.outerAxis == cmd.innerAxis,
             "MpuUnit: tensor_loop outer_axis must differ from inner_axis");
    panic_if(cmd.outerAxis == Axis::K || cmd.innerAxis == Axis::K,
             "MpuUnit: Phase 3 tensor_loop does not support K-axis expansion");
    panic_if(cmd.pingpongA || cmd.pingpongB || cmd.pingpongC,
             "MpuUnit: Phase 3 tensor_loop requires ping-pong disabled");
    panic_if(cmd.autoLoadCForAcc,
             "MpuUnit: Phase 3 tensor_loop requires auto_load_c_for_acc=0");
    panic_if((cmd.words[15] & 0xfff00000U) != 0,
             "MpuUnit: tensor_loop reserved step_cfg bits must be zero");
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

    Addr outer_a = parsedCmd.spmAddrA;
    Addr outer_b = parsedCmd.spmAddrB;
    Addr outer_c = parsedCmd.spmAddrC;

    for (uint32_t outer = 0; outer < parsedCmd.outerCount; ++outer) {
        Addr inner_a = outer_a;
        Addr inner_b = outer_b;
        Addr inner_c = outer_c;

        for (uint32_t inner = 0; inner < parsedCmd.innerCount; ++inner) {
            validateSpmAddress(inner_a, a_bytes, "tensor_loop A tile");
            validateSpmAddress(inner_b, b_bytes, "tensor_loop B tile");
            validateSpmAddress(inner_c, c_bytes, "tensor_loop C tile");

            IterationPlan plan;
            plan.validM = parsedCmd.validM;
            plan.validN = parsedCmd.validN;
            plan.validK = parsedCmd.validK;
            plan.subop = ComputeSubop::Matmul;
            plan.tensorAAddr = inner_a;
            plan.tensorBAddr = inner_b;
            plan.tensorCAddr = inner_c;
            plan.tensorA.assign(a_bytes, 0);
            plan.tensorB.assign(b_bytes, 0);
            plan.tensorC.assign(c_bytes, 0);
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
        plan.loadSpmAddr = parsedCmd.spmAddrA;
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
        plan.storeSpmAddr = parsedCmd.spmAddrC;
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

    switch (static_cast<Mode>(parsedCmd.mode)) {
      case Mode::Load:
        loadCount++;
        markBusyForFineCommand(parsedCmd);
        break;
      case Mode::Compute:
        computeCount++;
        markBusyForFineCommand(parsedCmd);
        break;
      case Mode::Store:
        storeCount++;
        markBusyForFineCommand(parsedCmd);
        break;
      case Mode::TensorLoop:
        tensorLoopCount++;
        break;
    }

    buildIterationPlans(exec);
    if (static_cast<Mode>(parsedCmd.mode) == Mode::TensorLoop) {
        tensorLoopExpandedTilesCount += iterationPlans.size();
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
        const SlotState &slot = slotByIndex(plan->loadSlotIndex);
        MemRequestDesc req;
        req.portId = 0;
        req.addr = plan->loadSpmAddr;
        req.size = tileBytesForSlotKind(slot.kind, plan->validM, plan->validN,
                                        plan->validK);
        reqs.push_back(req);
        pendingLoadTxns.emplace(token, PendingLoadTxn{
            exec.iteration, PendingLoadKind::SlotLoad, plan->loadSlotIndex});
        break;
      }
      case Mode::TensorLoop: {
        MemRequestDesc load_a;
        load_a.portId = 0;
        load_a.addr = plan->tensorAAddr;
        load_a.size = aTileBytes(plan->validM, plan->validK);
        reqs.push_back(load_a);
        pendingLoadTxns.emplace(token++, PendingLoadTxn{
            exec.iteration, PendingLoadKind::TensorA, -1});

        MemRequestDesc load_b;
        load_b.portId = 0;
        load_b.addr = plan->tensorBAddr;
        load_b.size = bTileBytes(plan->validK, plan->validN);
        reqs.push_back(load_b);
        pendingLoadTxns.emplace(token, PendingLoadTxn{
            exec.iteration, PendingLoadKind::TensorB, -1});
        break;
      }
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
        SlotState &slot = slotByIndex(pending.slotIndex);
        const size_t expected = tileBytesForSlotKind(slot.kind, parsedCmd.validM,
                                                     parsedCmd.validN,
                                                     parsedCmd.validK);
        panic_if(pkt->getSize() != expected,
                 "MpuUnit: load response size=%u expected=%zu",
                 pkt->getSize(), expected);
        std::fill(slot.bytes.begin(), slot.bytes.end(), 0);
        std::copy(data, data + expected, slot.bytes.begin());
        slot.valid = true;
        slot.dirty = false;
        slot.busy = false;
        setSlotShape(slot, parsedCmd);
        break;
      }
      case PendingLoadKind::TensorA: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        panic_if(pkt->getSize() != plan.tensorA.size(),
                 "MpuUnit: tensor_loop A response size=%u expected=%zu",
                 pkt->getSize(), plan.tensorA.size());
        std::copy(data, data + plan.tensorA.size(), plan.tensorA.begin());
        break;
      }
      case PendingLoadKind::TensorB: {
        IterationPlan &plan = iterationPlan(pending.iteration);
        panic_if(pkt->getSize() != plan.tensorB.size(),
                 "MpuUnit: tensor_loop B response size=%u expected=%zu",
                 pkt->getSize(), plan.tensorB.size());
        std::copy(data, data + plan.tensorB.size(), plan.tensorB.begin());
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

        std::vector<uint8_t> c_bytes = slot_c.bytes;
        runMatmul(slot_a.bytes, slot_b.bytes, c_bytes, plan.validM, plan.validN,
                  plan.validK, plan.subop == ComputeSubop::MatmulAcc);
        std::fill(slot_c.bytes.begin(), slot_c.bytes.end(), 0);
        std::copy(c_bytes.begin(), c_bytes.end(), slot_c.bytes.begin());
        slot_c.valid = true;
        slot_c.dirty = true;
        slot_c.busy = false;
        setSlotShape(slot_c, parsedCmd);
        clearSlotBusy(plan.computeSlotA);
        clearSlotBusy(plan.computeSlotB);

        if (plan.subop == ComputeSubop::Matmul) {
            matmulCountValue++;
            return matmulLatency(plan.validK);
        }

        matmulAccCountValue++;
        return matmulAccLatency(plan.validK);
      }
      case Mode::Store:
        return storeBaseLatency;
      case Mode::TensorLoop:
        runMatmul(plan.tensorA, plan.tensorB, plan.tensorC, plan.validM,
                  plan.validN, plan.validK, false);
        matmulCountValue++;
        return matmulLatency(plan.validK);
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
        const SlotState &slot = slotByIndex(plan->storeSlotIndex);
        MemRequestDesc req;
        req.portId = 0;
        req.addr = plan->storeSpmAddr;
        req.size = cTileBytes(plan->validM, plan->validN);
        req.data.assign(slot.bytes.begin(), slot.bytes.begin() + req.size);
        reqs.push_back(req);
        return;
      }
      case Mode::TensorLoop: {
        MemRequestDesc req;
        req.portId = 0;
        req.addr = plan->tensorCAddr;
        req.size = plan->tensorC.size();
        req.data = plan->tensorC;
        reqs.push_back(req);
        return;
      }
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
    (void)txn;
    (void)pkt;

    if (static_cast<Mode>(parsedCmd.mode) == Mode::Store) {
        SlotState &slot = slotByIndex(slotIndexForAddr(parsedCmd.localAddrC));
        slot.dirty = false;
        slot.busy = false;
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

} // namespace gem5
