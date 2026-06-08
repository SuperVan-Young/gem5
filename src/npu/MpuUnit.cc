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

#include "npu/MpuUnit.hh"

#include <algorithm>
#include <cstring>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/MpuUnit.hh"

namespace gem5
{

namespace
{

constexpr size_t ReadMaskWord = 1;
constexpr size_t WriteMaskWord = 2;
constexpr size_t RepetitionWord = 3;
constexpr size_t ReservedWord = 4;
constexpr size_t BufferWord = 5;
constexpr size_t MWord = 6;
constexpr size_t NWord = 7;
constexpr size_t KWord = 8;
constexpr size_t SpmAddrLoWord = 9;
constexpr size_t SpmAddrHiWord = 10;
constexpr size_t StrideWord = 11;
constexpr size_t FlagsWord = 12;
constexpr size_t ReservedWord13 = 13;
constexpr size_t ReservedWord14 = 14;
constexpr size_t ReservedWord15 = 15;
constexpr uint32_t FusedMatmulStrideUnitBytes = 16;
constexpr uint32_t FusedMatmulStrideMask = 0x3ff;
constexpr uint32_t ExecIssueQueueId = 0;
constexpr uint32_t PrefetchAIssueQueueId = 1;
constexpr uint32_t PrefetchBIssueQueueId = 2;
constexpr uint32_t StoreIssueQueueId = 3;
constexpr uint32_t MpuFlagAccumulate = 0x00000001U;
constexpr uint32_t MpuFlagLastKBlock = 0x00000002U;
constexpr uint32_t MpuFlagClearOutput = 0x00000004U;
constexpr uint32_t MpuFlagDrainToC = 0x00000008U;

bool
mpuFlagSet(uint32_t flags, uint32_t mask)
{
    return (flags & mask) != 0;
}

enum class FusedMatmulTokenKind : uint8_t
{
    ALoad = 0,
    BLoad = 1,
    CStore = 2,
};

uint64_t
makeFusedMatmulToken(FusedMatmulTokenKind kind, uint32_t row)
{
    return (static_cast<uint64_t>(kind) << 32) | row;
}

FusedMatmulTokenKind
fusedMatmulTokenKind(uint64_t token)
{
    return static_cast<FusedMatmulTokenKind>((token >> 32) & 0xff);
}

uint32_t
fusedMatmulTokenRow(uint64_t token)
{
    return static_cast<uint32_t>(token & 0xffffffffULL);
}

} // namespace

void
MpuUnit::ABBufferSlot::reset()
{
    state = BufferState::Empty;
    rows = cols = m = n = k = 0;
    data.clear();
}

void
MpuUnit::CBufferSlot::reset()
{
    state = BufferState::Empty;
    m = n = k = 0;
    data.clear();
}

void
MpuUnit::LoadedInputContext::reset()
{
    valid = false;
    bufferIndex = 0;
    rows = cols = m = n = k = 0;
}

void
MpuUnit::OutputStorage::reset()
{
    state = OutputStorageState::Empty;
    m = n = k = 0;
    readyTick = 0;
    data.clear();
}

MpuUnit::MpuStats::MpuStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(mvinCmdCount, statistics::units::Count::get(),
               "Number of completed mvin commands"),
      ADD_STAT(loadCmdCount, statistics::units::Count::get(),
               "Number of completed load commands"),
      ADD_STAT(computeCmdCount, statistics::units::Count::get(),
               "Number of completed compute commands"),
      ADD_STAT(drainCmdCount, statistics::units::Count::get(),
               "Number of completed drain commands"),
      ADD_STAT(mvoutCmdCount, statistics::units::Count::get(),
               "Number of completed mvout commands"),
      ADD_STAT(totalABytesIn, statistics::units::Byte::get(),
               "Total bytes read from SPM into A buffers"),
      ADD_STAT(totalBBytesIn, statistics::units::Byte::get(),
               "Total bytes read from SPM into B buffers"),
      ADD_STAT(totalCBytesOut, statistics::units::Byte::get(),
               "Total bytes written from C buffers to SPM"),
      ADD_STAT(totalOutputElementsDrained, statistics::units::Count::get(),
               "Total output elements drained into C buffers"),
      ADD_STAT(totalMacOps, statistics::units::Count::get(),
               "Total INT8xINT8->INT32 MAC operations"),
      ADD_STAT(busyCycles, statistics::units::Cycle::get(),
               "Cycles MPU spent processing a command"),
      ADD_STAT(idleCycles, statistics::units::Cycle::get(),
               "Cycles MPU spent idle"),
      ADD_STAT(stallCyclesWaitingForSpm, statistics::units::Cycle::get(),
               "Cycles attributable to waiting on SPM responses"),
      ADD_STAT(stallCyclesBufferHazard, statistics::units::Cycle::get(),
               "Cycles attributable to buffer hazards in the current stage"),
      ADD_STAT(stallCyclesOutputStorageUnavailable,
               statistics::units::Cycle::get(),
               "Cycles attributable to output storage unavailability"),
      ADD_STAT(stallCyclesDrainDestBusy, statistics::units::Cycle::get(),
               "Cycles attributable to busy drain destinations")
{
}

MpuUnit::MpuUnit(const MpuUnitParams &params)
    : SpecializedExecutionUnit(params),
      stats(this),
      arrayDim(params.array_dim),
      aBufferCapacityBytes(params.a_buffer_capacity_bytes),
      bBufferCapacityBytes(params.b_buffer_capacity_bytes),
      cBufferCapacityBytes(params.c_buffer_capacity_bytes),
      memUopQueueDepth(params.mem_uop_queue_depth),
      execUopQueueDepth(params.exec_uop_queue_depth),
      drainUopQueueDepth(params.drain_uop_queue_depth),
      mvinRequestLatency(params.mvin_request_latency),
      mvoutRequestLatency(params.mvout_request_latency),
      loadLatencyBase(params.load_latency_base),
      drainLatencyBase(params.drain_latency_base),
      numMpuMemSidePorts(params.num_mem_side_ports)
{
    fatal_if(macroCmdBytes != CacheLineBytes,
             "%s: MpuUnit requires 64-byte macro commands", name());
    fatal_if(arrayDim == 0,
             "%s: array_dim must be greater than zero", name());
    fatal_if(aBufferCapacityBytes == 0 || bBufferCapacityBytes == 0 ||
                 cBufferCapacityBytes == 0,
             "%s: buffer capacities must be greater than zero", name());
    fatal_if(cmdQueueDepth == 0 || cmdQueueDepth > 16,
             "%s: macro command queue depth must be in [1, 16]", name());
    fatal_if(memUopQueueDepth == 0 || memUopQueueDepth > 16,
             "%s: mem_uop_queue_depth must be in [1, 16]", name());
    fatal_if(execUopQueueDepth == 0 || execUopQueueDepth > 16,
             "%s: exec_uop_queue_depth must be in [1, 16]", name());
    fatal_if(drainUopQueueDepth == 0 || drainUopQueueDepth > 16,
             "%s: drain_uop_queue_depth must be in [1, 16]", name());
    fatal_if(memSidePorts.empty() || memSidePorts.size() > 3,
             "%s: MpuUnit requires one, two, or three mem_side ports",
             name());

    // Rebuild issue queues after base construction so MPU's override takes
    // effect; virtual dispatch does not apply during base-class construction.
    issueQueues = buildIssueQueues();

    busyStateKnown = true;
    busyState = false;
    busyStateChangeTick = 0;
    refreshScoreboard();
}

uint32_t
MpuUnit::extractWord(const std::vector<uint8_t> &cmd, size_t index) const
{
    panic_if((index + 1) * sizeof(uint32_t) > cmd.size(),
             "%s: command word %zu is out of range", name(), index);

    uint32_t word = 0;
    std::memcpy(&word, cmd.data() + (index * sizeof(uint32_t)), sizeof(word));
    return word;
}

MpuUnit::ParsedCmd
MpuUnit::parseCommand(const std::vector<uint8_t> &cmd) const
{
    panic_if(cmd.size() != CacheLineBytes,
             "%s: expected 64-byte command, got %zu bytes",
             name(), cmd.size());

    ParsedCmd parsed;
    parsed.header = parseCmdFields(extractCmdWord(cmd));
    parsed.dataType = (parsed.header.opCode >> 5) & 0x7;
    parsed.kind = static_cast<CmdKind>(parsed.header.opCode & 0x1f);

    const uint32_t bufferWord = extractWord(cmd, BufferWord);
    parsed.bufferKind = static_cast<BufferKind>(bufferWord & 0x3U);
    parsed.bufferIndex = (bufferWord >> 2) & 0x1U;
    parsed.m = extractWord(cmd, MWord);
    parsed.n = extractWord(cmd, NWord);
    parsed.k = extractWord(cmd, KWord);

    if (parsed.kind == CmdKind::FusedMatmul) {
        const uint32_t tile_word = bufferWord;
        const uint32_t stride_word = extractWord(cmd, ReservedWord15);

        parsed.bufferKind = BufferKind::Reserved;
        parsed.bufferIndex = 0;
        parsed.tileM = tile_word & 0xffU;
        parsed.tileN = (tile_word >> 8) & 0xffU;
        parsed.tileK = (tile_word >> 16) & 0xffU;
        parsed.aBase =
            (static_cast<Addr>(extractWord(cmd, SpmAddrHiWord)) << 32) |
            static_cast<Addr>(extractWord(cmd, SpmAddrLoWord));
        parsed.bBase =
            (static_cast<Addr>(extractWord(cmd, ReservedWord13)) << 32) |
            static_cast<Addr>(extractWord(cmd, StrideWord));
        parsed.cBase =
            (static_cast<Addr>(extractWord(cmd, ReservedWord14)) << 32) |
            static_cast<Addr>(extractWord(cmd, FlagsWord));
        parsed.aRowStrideBytes =
            (stride_word & FusedMatmulStrideMask) *
            FusedMatmulStrideUnitBytes;
        parsed.bRowStrideBytes =
            ((stride_word >> 10) & FusedMatmulStrideMask) *
            FusedMatmulStrideUnitBytes;
        parsed.cRowStrideBytes =
            ((stride_word >> 20) & 0xfffU) * FusedMatmulStrideUnitBytes;
        return parsed;
    }

    parsed.spmAddr =
        (static_cast<Addr>(extractWord(cmd, SpmAddrHiWord)) << 32) |
        static_cast<Addr>(extractWord(cmd, SpmAddrLoWord));
    parsed.strideBytes = extractWord(cmd, StrideWord);
    parsed.flags = extractWord(cmd, FlagsWord);
    return parsed;
}

void
MpuUnit::validateSpmWindow(const ParsedCmd &cmd) const
{
    const uint32_t rows = expectedRows(cmd);
    const uint32_t rowBytes = expectedRowBytes(cmd);

    panic_if(cmd.spmAddr < SpmBase || cmd.spmAddr > SpmEnd,
             "%s: SPM address %#llx is outside the SPM window",
             name(), static_cast<unsigned long long>(cmd.spmAddr));
    panic_if(cmd.strideBytes < rowBytes,
             "%s: stride_bytes=%u is smaller than row_bytes=%u",
             name(), cmd.strideBytes, rowBytes);

    const unsigned long long totalEnd =
        static_cast<unsigned long long>(cmd.spmAddr) +
        static_cast<unsigned long long>(rows - 1) * cmd.strideBytes +
        rowBytes;
    panic_if(totalEnd - 1 > SpmEnd,
             "%s: SPM window [%#llx, %#llx] exceeds SPM end %#llx",
             name(),
             static_cast<unsigned long long>(cmd.spmAddr),
             totalEnd - 1,
             static_cast<unsigned long long>(SpmEnd));
}

uint32_t
MpuUnit::expectedRows(const ParsedCmd &cmd) const
{
    switch (cmd.bufferKind) {
      case BufferKind::A:
        return cmd.m;
      case BufferKind::B:
        return cmd.k;
      case BufferKind::C:
        return cmd.m;
      case BufferKind::Reserved:
        break;
    }

    panic("%s: expectedRows reached reserved buffer kind", name());
}

uint32_t
MpuUnit::expectedCols(const ParsedCmd &cmd) const
{
    switch (cmd.bufferKind) {
      case BufferKind::A:
        return cmd.k;
      case BufferKind::B:
        return cmd.n;
      case BufferKind::C:
        return cmd.n;
      case BufferKind::Reserved:
        break;
    }

    panic("%s: expectedCols reached reserved buffer kind", name());
}

uint32_t
MpuUnit::expectedRowBytes(const ParsedCmd &cmd) const
{
    switch (cmd.bufferKind) {
      case BufferKind::A:
        return cmd.k;
      case BufferKind::B:
        return cmd.n;
      case BufferKind::C:
        return cmd.n * sizeof(int32_t);
      case BufferKind::Reserved:
        break;
    }

    panic("%s: expectedRowBytes reached reserved buffer kind", name());
}

uint32_t
MpuUnit::requiredBytes(const ParsedCmd &cmd) const
{
    return expectedRows(cmd) * expectedRowBytes(cmd);
}

bool
MpuUnit::isContiguousSpmWindow(const ParsedCmd &cmd) const
{
    return cmd.strideBytes == expectedRowBytes(cmd);
}

void
MpuUnit::validateMvin(const ParsedCmd &cmd) const
{
    panic_if(cmd.bufferKind == BufferKind::C,
             "%s: mvin(C) is not supported in MPU v1.1", name());
    panic_if(cmd.bufferKind == BufferKind::Reserved,
             "%s: mvin requires an A or B destination buffer", name());

    const uint32_t needed = requiredBytes(cmd);
    if (cmd.bufferKind == BufferKind::A) {
        panic_if(needed > aBufferCapacityBytes,
                 "%s: A tile bytes=%u exceed A buffer capacity=%u",
                 name(), needed, aBufferCapacityBytes);
        panic_if(aBuffers[cmd.bufferIndex].state != BufferState::Empty,
                 "%s: A%u is not available for mvin", name(),
                 cmd.bufferIndex);
    } else {
        panic_if(needed > bBufferCapacityBytes,
                 "%s: B tile bytes=%u exceed B buffer capacity=%u",
                 name(), needed, bBufferCapacityBytes);
        panic_if(bBuffers[cmd.bufferIndex].state != BufferState::Empty,
                 "%s: B%u is not available for mvin", name(),
                 cmd.bufferIndex);
    }

    validateSpmWindow(cmd);
}

void
MpuUnit::validateLoadStatic(const ParsedCmd &cmd) const
{
    panic_if(cmd.bufferKind == BufferKind::C,
             "%s: load(C) is not supported in MPU v1.1", name());
    panic_if(cmd.bufferKind == BufferKind::Reserved,
             "%s: load requires an A or B source buffer", name());
}

bool
MpuUnit::loadReady(const ParsedCmd &cmd) const
{
    const ABBufferSlot &slot = selectedABuffer(cmd.bufferKind,
                                               cmd.bufferIndex);
    if (slot.state == BufferState::Empty ||
        slot.state == BufferState::LoadingFromSpm) {
        return false;
    }

    panic_if(slot.state != BufferState::Full,
             "%s: load requires a FULL source buffer", name());
    panic_if(slot.rows != expectedRows(cmd) || slot.cols != expectedCols(cmd),
             "%s: load dimensions (%u,%u) do not match buffer metadata (%u,%u)",
             name(), expectedRows(cmd), expectedCols(cmd), slot.rows,
             slot.cols);
    if (cmd.bufferKind == BufferKind::A) {
        panic_if(loadedA.valid,
                 "%s: loaded A source is already occupied", name());
    } else {
        panic_if(loadedB.valid,
                 "%s: loaded B source is already occupied", name());
    }
    return true;
}

void
MpuUnit::validateCompute(const ParsedCmd &cmd) const
{
    panic_if(!loadedA.valid || !loadedB.valid,
             "%s: compute requires one loaded A tile and one loaded B tile",
             name());
    panic_if(outputStorage.state != OutputStorageState::Empty,
             "%s: compute requires output storage to be empty", name());
    panic_if(loadedA.m != cmd.m || loadedA.k != cmd.k,
             "%s: loaded A metadata (%u,%u) does not match compute (%u,%u)",
             name(), loadedA.m, loadedA.k, cmd.m, cmd.k);
    panic_if(loadedB.k != cmd.k || loadedB.n != cmd.n,
             "%s: loaded B metadata (%u,%u) does not match compute (%u,%u)",
             name(), loadedB.k, loadedB.n, cmd.k, cmd.n);
}

void
MpuUnit::validateComputeFused(const ParsedCmd &cmd) const
{
    panic_if(cmd.bufferKind != BufferKind::Reserved,
             "%s: fused compute expects reserved buffer kind", name());

    const bool accumulate = mpuFlagSet(cmd.flags, MpuFlagAccumulate);
    const bool clear = mpuFlagSet(cmd.flags, MpuFlagClearOutput);
    const bool drain_to_c = mpuFlagSet(cmd.flags, MpuFlagDrainToC);
    panic_if(accumulate && clear,
             "%s: fused compute cannot both accumulate and clear output",
             name());
    panic_if(drain_to_c && !mpuFlagSet(cmd.flags, MpuFlagLastKBlock),
             "%s: fused compute can drain to C only on the last K block",
             name());
}

void
MpuUnit::validateFusedMatmul(const ParsedCmd &cmd) const
{
    panic_if(cmd.tileM == 0 || cmd.tileN == 0 || cmd.tileK == 0,
             "%s: fused matmul requires non-zero tile shape", name());
    panic_if(cmd.tileM > arrayDim || cmd.tileN > arrayDim,
             "%s: fused matmul tile (%u,%u) exceeds array_dim=%u",
             name(), cmd.tileM, cmd.tileN, arrayDim);
    panic_if(cmd.k != cmd.tileK,
             "%s: fused matmul currently requires k == tile_k (%u != %u)",
             name(), cmd.k, cmd.tileK);

    const uint32_t a_row_bytes = cmd.tileK;
    const uint32_t b_row_bytes = cmd.tileN;
    const uint32_t c_row_bytes = cmd.tileN * sizeof(int32_t);
    panic_if(cmd.aRowStrideBytes < a_row_bytes,
             "%s: fused matmul A stride=%u is smaller than row_bytes=%u",
             name(), cmd.aRowStrideBytes, a_row_bytes);
    panic_if(cmd.bRowStrideBytes < b_row_bytes,
             "%s: fused matmul B stride=%u is smaller than row_bytes=%u",
             name(), cmd.bRowStrideBytes, b_row_bytes);
    panic_if(cmd.cRowStrideBytes < c_row_bytes,
             "%s: fused matmul C stride=%u is smaller than row_bytes=%u",
             name(), cmd.cRowStrideBytes, c_row_bytes);
    panic_if(cmd.aRowStrideBytes != a_row_bytes ||
                 cmd.bRowStrideBytes != b_row_bytes ||
                 cmd.cRowStrideBytes != c_row_bytes,
             "%s: fused matmul currently requires contiguous A/B/C matrices",
             name());

    auto validate_window = [this](Addr base, uint32_t rows,
                                  uint32_t stride, uint32_t row_bytes,
                                  const char *label) {
        panic_if(base < SpmBase || base > SpmEnd,
                 "%s: fused matmul %s base %#llx is outside SPM",
                 name(), label, static_cast<unsigned long long>(base));
        const unsigned long long total_end =
            static_cast<unsigned long long>(base) +
            static_cast<unsigned long long>(rows - 1) * stride + row_bytes;
        panic_if(total_end - 1 > SpmEnd,
                 "%s: fused matmul %s window [%#llx, %#llx] exceeds SPM",
                 name(), label, static_cast<unsigned long long>(base),
                 total_end - 1);
    };

    validate_window(cmd.aBase, cmd.m, cmd.aRowStrideBytes, a_row_bytes, "A");
    validate_window(cmd.bBase, cmd.k, cmd.bRowStrideBytes, b_row_bytes, "B");
    validate_window(cmd.cBase, cmd.m, cmd.cRowStrideBytes, c_row_bytes, "C");
}

bool
MpuUnit::fusedComputeReady(const ParsedCmd &cmd) const
{
    const uint8_t index = cmd.bufferIndex;
    const ABBufferSlot &a = aBuffers[index];
    const ABBufferSlot &b = bBuffers[index];

    if (a.state == BufferState::Empty ||
        a.state == BufferState::LoadingFromSpm ||
        b.state == BufferState::Empty ||
        b.state == BufferState::LoadingFromSpm) {
        return false;
    }

    panic_if(a.state != BufferState::Full,
             "%s: fused compute requires FULL A%u", name(), index);
    panic_if(b.state != BufferState::Full,
             "%s: fused compute requires FULL B%u", name(), index);
    panic_if(a.rows != cmd.m || a.cols != cmd.k,
             "%s: fused A metadata (%u,%u) does not match (%u,%u)",
             name(), a.rows, a.cols, cmd.m, cmd.k);
    panic_if(b.rows != cmd.k || b.cols != cmd.n,
             "%s: fused B metadata (%u,%u) does not match (%u,%u)",
             name(), b.rows, b.cols, cmd.k, cmd.n);
    panic_if(a.data.size() != static_cast<size_t>(cmd.m) * cmd.k,
             "%s: A buffer payload size mismatch for fused compute", name());
    panic_if(b.data.size() != static_cast<size_t>(cmd.k) * cmd.n,
             "%s: B buffer payload size mismatch for fused compute", name());

    const bool accumulate = mpuFlagSet(cmd.flags, MpuFlagAccumulate);
    const bool drain_to_c = mpuFlagSet(cmd.flags, MpuFlagDrainToC);
    if (!accumulate) {
        panic_if(outputStorage.state != OutputStorageState::Empty,
                 "%s: fused compute first block requires empty output",
                 name());
    } else {
        panic_if(outputStorage.state == OutputStorageState::Empty,
                 "%s: fused accumulate requires existing output", name());
        panic_if(outputStorage.m != cmd.m || outputStorage.n != cmd.n,
                 "%s: fused accumulate shape mismatch", name());
    }
    if (drain_to_c) {
        const CBufferSlot &c = selectedCBuffer(index);
        panic_if(c.state != BufferState::Empty,
                 "%s: fused compute drain-to-C requires EMPTY C%u",
                 name(), index);
        panic_if(static_cast<uint64_t>(cmd.m) * cmd.n * sizeof(int32_t) >
                     cBufferCapacityBytes,
                 "%s: fused drained C tile bytes exceed C buffer capacity",
                 name());
    }

    return true;
}

void
MpuUnit::validateDrain(const ParsedCmd &cmd) const
{
    panic_if(cmd.bufferKind != BufferKind::C,
             "%s: drain requires a C destination buffer", name());
    panic_if(outputStorage.state != OutputStorageState::ReadyToDrain,
             "%s: drain requires output storage to be READY_TO_DRAIN", name());
    panic_if(outputStorage.m != cmd.m || outputStorage.n != cmd.n,
             "%s: drain dimensions (%u,%u) do not match output storage (%u,%u)",
             name(), cmd.m, cmd.n, outputStorage.m, outputStorage.n);
    panic_if(cBuffers[cmd.bufferIndex].state != BufferState::Empty,
             "%s: C%u is not available as a drain destination", name(),
             cmd.bufferIndex);
    panic_if(requiredBytes(cmd) > cBufferCapacityBytes,
             "%s: drained C tile bytes=%u exceed C buffer capacity=%u",
             name(), requiredBytes(cmd), cBufferCapacityBytes);
}

void
MpuUnit::validateMvout(const ParsedCmd &cmd) const
{
    panic_if(cmd.bufferKind != BufferKind::C,
             "%s: mvout requires a C source buffer", name());

    const CBufferSlot &slot = selectedCBuffer(cmd.bufferIndex);
    panic_if(slot.state != BufferState::Full,
             "%s: mvout requires a FULL C source buffer", name());
    panic_if(slot.m != cmd.m || slot.n != cmd.n,
             "%s: mvout dimensions (%u,%u) do not match C buffer metadata (%u,%u)",
             name(), cmd.m, cmd.n, slot.m, slot.n);
    validateSpmWindow(cmd);
}

void
MpuUnit::validateCommand(const std::vector<uint8_t> &rawCmd,
                         const ParsedCmd &cmd) const
{
    panic_if(cmd.header.deviceType != MpuDeviceType,
             "%s: unexpected device_type=%u for MPU command", name(),
             cmd.header.deviceType);
    panic_if(cmd.dataType != Int8DataType,
             "%s: unsupported MPU data_type=%u", name(), cmd.dataType);
    panic_if(cmd.bufferIndex > 1,
             "%s: invalid buffer_index=%u", name(), cmd.bufferIndex);
    panic_if(cmd.m == 0 || cmd.n == 0 || cmd.k == 0,
             "%s: illegal dimensions m=%u n=%u k=%u", name(),
             cmd.m, cmd.n, cmd.k);
    panic_if(cmd.kind != CmdKind::FusedMatmul &&
                 (cmd.m > arrayDim || cmd.n > arrayDim),
             "%s: dimensions m=%u n=%u exceed array_dim=%u", name(),
             cmd.m, cmd.n, arrayDim);

    panic_if(extractWord(rawCmd, ReadMaskWord) != 0,
             "%s: MPU commands require readMask == 0", name());
    panic_if(extractWord(rawCmd, WriteMaskWord) != 0,
             "%s: MPU commands require writeMask == 0", name());
    panic_if(extractWord(rawCmd, RepetitionWord) > 1,
             "%s: MPU commands only support repetition <= 1 "
             "in the current stage",
             name());
    panic_if(extractWord(rawCmd, ReservedWord) != 0,
             "%s: MPU commands require reserved word 4 == 0", name());
    if (cmd.kind == CmdKind::FusedMatmul) {
        panic_if(extractWord(rawCmd, ReservedWord15) == 0,
                 "%s: fused matmul requires packed stride word",
                 name());
    } else {
        const uint32_t allowedFlags =
            cmd.kind == CmdKind::ComputeFused ?
                (MpuFlagAccumulate | MpuFlagLastKBlock |
                 MpuFlagClearOutput | MpuFlagDrainToC) :
                0U;
        panic_if((cmd.flags & ~allowedFlags) != 0 ||
                     extractWord(rawCmd, ReservedWord13) != 0 ||
                     extractWord(rawCmd, ReservedWord14) != 0 ||
                     extractWord(rawCmd, ReservedWord15) != 0,
                 "%s: MPU command reserved words must be zero", name());
    }

    switch (cmd.kind) {
      case CmdKind::Mvin:
        validateMvin(cmd);
        return;
      case CmdKind::Mvout:
        validateMvout(cmd);
        return;
      case CmdKind::Load:
        validateLoadStatic(cmd);
        return;
      case CmdKind::Compute:
        validateCompute(cmd);
        return;
      case CmdKind::ComputeFused:
        validateComputeFused(cmd);
        return;
      case CmdKind::FusedMatmul:
        validateFusedMatmul(cmd);
        return;
      case CmdKind::Drain:
        validateDrain(cmd);
        return;
    }

    panic("%s: unsupported MPU command kind %#x", name(),
          static_cast<unsigned>(cmd.kind));
}

void
MpuUnit::resetCommandStructures()
{
    // The SEU base owns the real per-macro uop queues. MPU's local queue
    // structures are observational only and may remain populated while
    // another issue queue is active.
}

void
MpuUnit::refreshScoreboard()
{
    scoreboard.loadedAReady = loadedA.valid;
    scoreboard.loadedBReady = loadedB.valid;
    scoreboard.outputStorageActive =
        outputStorage.state == OutputStorageState::Active;
    scoreboard.outputStorageReady =
        outputStorage.state == OutputStorageState::ReadyToDrain;
    for (size_t i = 0; i < scoreboard.cDrainReserved.size(); ++i) {
        scoreboard.cDrainReserved[i] =
            cBuffers[i].state == BufferState::DrainingToBuffer;
        scoreboard.cWritebackBusy[i] =
            cBuffers[i].state == BufferState::WritingToSpm;
    }
}

void
MpuUnit::updateBusyAccounting(bool now_busy)
{
    if (!busyStateKnown) {
        busyStateKnown = true;
        busyState = now_busy;
        busyStateChangeTick = curTick();
        return;
    }

    if (busyState == now_busy) {
        return;
    }

    const uint64_t delta_cycles =
        static_cast<uint64_t>(ticksToCycles(curTick() - busyStateChangeTick));
    if (busyState) {
        stats.busyCycles += delta_cycles;
    } else {
        stats.idleCycles += delta_cycles;
    }

    busyState = now_busy;
    busyStateChangeTick = curTick();
}

uint64_t
MpuUnit::elapsedCyclesSince(Tick start) const
{
    return static_cast<uint64_t>(ticksToCycles(curTick() - start));
}

uint64_t
MpuUnit::currentBusyCycles() const
{
    uint64_t total = stats.busyCycles.value();
    if (busyStateKnown && busyState) {
        total += static_cast<uint64_t>(
            ticksToCycles(curTick() - busyStateChangeTick));
    }
    return total;
}

uint64_t
MpuUnit::currentIdleCycles() const
{
    uint64_t total = stats.idleCycles.value();
    if (busyStateKnown && !busyState) {
        total += static_cast<uint64_t>(
            ticksToCycles(curTick() - busyStateChangeTick));
    }
    return total;
}

MpuUnit::ABBufferSlot &
MpuUnit::selectedABuffer(BufferKind kind, uint8_t index)
{
    panic_if(index > 1, "%s: invalid A/B buffer index %u", name(), index);
    if (kind == BufferKind::A) {
        return aBuffers[index];
    }
    if (kind == BufferKind::B) {
        return bBuffers[index];
    }

    panic("%s: selectedABuffer requires an A or B buffer", name());
}

const MpuUnit::ABBufferSlot &
MpuUnit::selectedABuffer(BufferKind kind, uint8_t index) const
{
    panic_if(index > 1, "%s: invalid A/B buffer index %u", name(), index);
    if (kind == BufferKind::A) {
        return aBuffers[index];
    }
    if (kind == BufferKind::B) {
        return bBuffers[index];
    }

    panic("%s: selectedABuffer requires an A or B buffer", name());
}

MpuUnit::CBufferSlot &
MpuUnit::selectedCBuffer(uint8_t index)
{
    panic_if(index > 1, "%s: invalid C buffer index %u", name(), index);
    return cBuffers[index];
}

const MpuUnit::CBufferSlot &
MpuUnit::selectedCBuffer(uint8_t index) const
{
    panic_if(index > 1, "%s: invalid C buffer index %u", name(), index);
    return cBuffers[index];
}

void
MpuUnit::beginMemWindow(MpuMacroRuntime &runtime)
{
    runtime.memWindow.active = true;
    runtime.memWindow.startTick = curTick();
    runtime.memWindow.lastRespTick = curTick();
}

void
MpuUnit::observeMemResponse(MpuMacroRuntime &runtime)
{
    if (!runtime.memWindow.active) {
        return;
    }
    runtime.memWindow.lastRespTick = curTick();
}

void
MpuUnit::finalizeMemWindow(MpuMacroRuntime &runtime)
{
    if (!runtime.memWindow.active) {
        return;
    }

    if (runtime.memWindow.lastRespTick >= runtime.memWindow.startTick) {
        stats.stallCyclesWaitingForSpm += static_cast<uint64_t>(ticksToCycles(
            runtime.memWindow.lastRespTick - runtime.memWindow.startTick));
    }
    runtime.memWindow = PendingMemWindow{};
}

PortID
MpuUnit::mvinPortId(BufferKind kind) const
{
    if (kind == BufferKind::B && numMpuMemSidePorts > 2) {
        return mvinBPortId();
    }
    return mvinAPortId();
}

PortID
MpuUnit::mvinAPortId() const
{
    return 0;
}

PortID
MpuUnit::mvinBPortId() const
{
    return numMpuMemSidePorts > 2 ? 1 : 0;
}

PortID
MpuUnit::mvoutPortId() const
{
    if (numMpuMemSidePorts > 2) {
        return 2;
    }
    return numMpuMemSidePorts > 1 ? 1 : 0;
}

MpuUnit::MpuMacroRuntime &
MpuUnit::runtimeFor(uint64_t macroCmdId)
{
    auto it = macroRuntimes.find(macroCmdId);
    panic_if(it == macroRuntimes.end(),
             "%s: missing MPU macro runtime for macro %llu",
             name(), static_cast<unsigned long long>(macroCmdId));
    return it->second;
}

const MpuUnit::MpuMacroRuntime &
MpuUnit::runtimeFor(uint64_t macroCmdId) const
{
    auto it = macroRuntimes.find(macroCmdId);
    panic_if(it == macroRuntimes.end(),
             "%s: missing MPU macro runtime for macro %llu",
             name(), static_cast<unsigned long long>(macroCmdId));
    return it->second;
}

void
MpuUnit::transitionABufferToFull(ABBufferSlot &slot, const ParsedCmd &cmd)
{
    slot.state = BufferState::Full;
    slot.rows = expectedRows(cmd);
    slot.cols = expectedCols(cmd);
    slot.m = cmd.m;
    slot.n = cmd.n;
    slot.k = cmd.k;
}

void
MpuUnit::transitionABufferToLoaded(ABBufferSlot &slot, const ParsedCmd &cmd)
{
    slot.state = BufferState::LoadedToInput;

    LoadedInputContext ctx;
    ctx.valid = true;
    ctx.bufferIndex = cmd.bufferIndex;
    ctx.rows = slot.rows;
    ctx.cols = slot.cols;
    ctx.m = slot.m;
    ctx.n = slot.n;
    ctx.k = slot.k;

    if (cmd.bufferKind == BufferKind::A) {
        loadedA = ctx;
    } else {
        loadedB = ctx;
    }
}

void
MpuUnit::transitionCBufferToFull(CBufferSlot &slot, const ParsedCmd &cmd)
{
    slot.state = BufferState::Full;
    slot.m = cmd.m;
    slot.n = cmd.n;
    slot.k = cmd.k;
}

void
MpuUnit::clearLoadedContext(BufferKind kind)
{
    if (kind == BufferKind::A) {
        loadedA.reset();
        return;
    }
    if (kind == BufferKind::B) {
        loadedB.reset();
        return;
    }
}

void
MpuUnit::releaseConsumedInputBuffers()
{
    // Current-stage engineering policy: after a successful compute, the
    // consumed A/B buffers are automatically released so the single-command
    // implementation remains reusable without introducing an explicit release
    // command. This is not the permanent MPU architecture contract.
    if (loadedA.valid) {
        aBuffers[loadedA.bufferIndex].reset();
        loadedA.reset();
    }
    if (loadedB.valid) {
        bBuffers[loadedB.bufferIndex].reset();
        loadedB.reset();
    }
}

void
MpuUnit::performCompute(const ParsedCmd &cmd)
{
    const ABBufferSlot &a = aBuffers[loadedA.bufferIndex];
    const ABBufferSlot &b = bBuffers[loadedB.bufferIndex];

    panic_if(a.data.size() != static_cast<size_t>(cmd.m) * cmd.k,
             "%s: A buffer payload size mismatch for compute", name());
    panic_if(b.data.size() != static_cast<size_t>(cmd.k) * cmd.n,
             "%s: B buffer payload size mismatch for compute", name());

    outputStorage.data.assign(static_cast<size_t>(cmd.m) * cmd.n, 0);
    outputStorage.m = cmd.m;
    outputStorage.n = cmd.n;
    outputStorage.k = cmd.k;

    for (uint32_t row = 0; row < cmd.m; ++row) {
        for (uint32_t col = 0; col < cmd.n; ++col) {
            int32_t acc = 0;
            for (uint32_t depth = 0; depth < cmd.k; ++depth) {
                const int32_t a_val =
                    static_cast<int32_t>(a.data[row * cmd.k + depth]);
                const int32_t b_val =
                    static_cast<int32_t>(b.data[depth * cmd.n + col]);
                acc += a_val * b_val;
            }
            outputStorage.data[row * cmd.n + col] = acc;
        }
    }

    outputStorage.state = OutputStorageState::ReadyToDrain;
    stats.totalMacOps +=
        static_cast<uint64_t>(cmd.m) * cmd.n * cmd.k;
    releaseConsumedInputBuffers();
}

void
MpuUnit::performComputeFused(const ParsedCmd &cmd)
{
    const uint8_t index = cmd.bufferIndex;
    const ABBufferSlot &a = aBuffers[index];
    const ABBufferSlot &b = bBuffers[index];
    const bool accumulate = mpuFlagSet(cmd.flags, MpuFlagAccumulate);
    const bool last = mpuFlagSet(cmd.flags, MpuFlagLastKBlock);
    const bool drain_to_c = mpuFlagSet(cmd.flags, MpuFlagDrainToC);

    if (!accumulate || outputStorage.state == OutputStorageState::Empty) {
        outputStorage.data.assign(static_cast<size_t>(cmd.m) * cmd.n, 0);
        outputStorage.m = cmd.m;
        outputStorage.n = cmd.n;
        outputStorage.k = 0;
    }

    for (uint32_t row = 0; row < cmd.m; ++row) {
        for (uint32_t col = 0; col < cmd.n; ++col) {
            int32_t acc = outputStorage.data[row * cmd.n + col];
            for (uint32_t depth = 0; depth < cmd.k; ++depth) {
                const int32_t a_val =
                    static_cast<int32_t>(a.data[row * cmd.k + depth]);
                const int32_t b_val =
                    static_cast<int32_t>(b.data[depth * cmd.n + col]);
                acc += a_val * b_val;
            }
            outputStorage.data[row * cmd.n + col] = acc;
        }
    }

    outputStorage.k += cmd.k;
    outputStorage.readyTick = curTick();
    stats.totalMacOps +=
        static_cast<uint64_t>(cmd.m) * cmd.n * cmd.k;

    if (drain_to_c) {
        CBufferSlot &slot = selectedCBuffer(index);
        slot.data = outputStorage.data;
        transitionCBufferToFull(slot, cmd);
        stats.totalOutputElementsDrained +=
            static_cast<uint64_t>(cmd.m) * cmd.n;
        outputStorage.reset();
    } else {
        outputStorage.state = last ?
            OutputStorageState::ReadyToDrain :
            OutputStorageState::Active;
    }

    aBuffers[index].reset();
    bBuffers[index].reset();
}

void
MpuUnit::performDrain(const ParsedCmd &cmd)
{
    CBufferSlot &slot = selectedCBuffer(cmd.bufferIndex);
    slot.data = outputStorage.data;
    transitionCBufferToFull(slot, cmd);
    stats.totalOutputElementsDrained +=
        static_cast<uint64_t>(cmd.m) * cmd.n;
    outputStorage.reset();
}

std::vector<uint8_t>
MpuUnit::serializeCRow(const CBufferSlot &slot, uint32_t row) const
{
    const size_t rowElems = slot.n;
    std::vector<uint8_t> bytes(rowElems * sizeof(int32_t), 0);
    std::memcpy(bytes.data(), slot.data.data() + row * rowElems, bytes.size());
    return bytes;
}

std::vector<uint8_t>
MpuUnit::serializeCTile(const CBufferSlot &slot) const
{
    std::vector<uint8_t> bytes(slot.data.size() * sizeof(int32_t), 0);
    std::memcpy(bytes.data(), slot.data.data(), bytes.size());
    return bytes;
}

void
MpuUnit::pushQueueEntry(uint64_t macroCmdId, const ParsedCmd &cmd)
{
    QueueEntry entry{cmd.kind, cmd.bufferKind, cmd.bufferIndex,
                     cmd.m, cmd.n, cmd.k};

    panic_if(!macroCmdFifo.emplace(macroCmdId, entry).second,
             "%s: duplicate macro FIFO entry for macro %llu",
             name(), static_cast<unsigned long long>(macroCmdId));
    panic_if(macroCmdFifo.size() > cmdQueueDepth,
             "%s: macro FIFO overflow", name());

    switch (cmd.kind) {
      case CmdKind::Mvin:
      case CmdKind::Mvout:
        panic_if(memUopQueue.size() >= memUopQueueDepth,
                 "%s: mem uop queue overflow", name());
        memUopQueue.emplace(macroCmdId, entry);
        break;
      case CmdKind::Load:
      case CmdKind::Compute:
      case CmdKind::ComputeFused:
      case CmdKind::FusedMatmul:
        panic_if(execUopQueue.size() >= execUopQueueDepth,
                 "%s: exec uop queue overflow", name());
        execUopQueue.emplace(macroCmdId, entry);
        break;
      case CmdKind::Drain:
        panic_if(drainUopQueue.size() >= drainUopQueueDepth,
                 "%s: drain uop queue overflow", name());
        drainUopQueue.emplace(macroCmdId, entry);
        break;
    }
}

void
MpuUnit::popQueueEntry(uint64_t macroCmdId, const ParsedCmd &cmd)
{
    switch (cmd.kind) {
      case CmdKind::Mvin:
      case CmdKind::Mvout:
        memUopQueue.erase(macroCmdId);
        break;
      case CmdKind::Load:
      case CmdKind::Compute:
      case CmdKind::ComputeFused:
      case CmdKind::FusedMatmul:
        execUopQueue.erase(macroCmdId);
        break;
      case CmdKind::Drain:
        drainUopQueue.erase(macroCmdId);
        break;
    }

    macroCmdFifo.erase(macroCmdId);
}

void
MpuUnit::appendMvinRowUop(MacroCmdContext &macroCmd,
                          MpuMacroRuntime &runtime)
{
    const ParsedCmd &cmd = runtime.parsed;
    const uint32_t row = runtime.nextMemRow;
    panic_if(row >= expectedRows(cmd),
             "%s: mvin row index %u is out of range", name(), row);
    if (row == 0) {
        beginMemWindow(runtime);
    }

    if (isContiguousSpmWindow(cmd)) {
        appendLoadUop(macroCmd, cmd.spmAddr, requiredBytes(cmd));
        macroCmd.uopQueue.back().portId = mvinPortId(cmd.bufferKind);
        macroCmd.uopQueue.back().token = 0;
        runtime.nextMemRow = expectedRows(cmd);
        return;
    }

    appendLoadUop(macroCmd,
                  cmd.spmAddr + static_cast<Addr>(row) * cmd.strideBytes,
                  expectedRowBytes(cmd));
    macroCmd.uopQueue.back().portId = mvinPortId(cmd.bufferKind);
    macroCmd.uopQueue.back().token = row;
    runtime.nextMemRow++;
}

void
MpuUnit::appendMvoutRowUop(MacroCmdContext &macroCmd,
                           MpuMacroRuntime &runtime)
{
    const ParsedCmd &cmd = runtime.parsed;
    const uint32_t row = runtime.nextMemRow;
    panic_if(row >= cmd.m, "%s: mvout row index %u is out of range",
             name(), row);
    if (row == 0) {
        beginMemWindow(runtime);
    }

    if (isContiguousSpmWindow(cmd)) {
        appendStoreUop(macroCmd, cmd.spmAddr, requiredBytes(cmd),
                       serializeCTile(selectedCBuffer(cmd.bufferIndex)));
        macroCmd.uopQueue.back().portId = mvoutPortId();
        macroCmd.uopQueue.back().token = 0;
        runtime.nextMemRow = cmd.m;
        return;
    }

    appendStoreUop(macroCmd,
                   cmd.spmAddr + static_cast<Addr>(row) * cmd.strideBytes,
                   cmd.n * sizeof(int32_t),
                   serializeCRow(selectedCBuffer(cmd.bufferIndex), row));
    macroCmd.uopQueue.back().portId = mvoutPortId();
    macroCmd.uopQueue.back().token = row;
    runtime.nextMemRow++;
}

void
MpuUnit::appendLoadProgressUop(MacroCmdContext &macroCmd,
                               MpuMacroRuntime &runtime)
{
    const ParsedCmd &cmd = runtime.parsed;
    if (loadReady(cmd)) {
        runtime.pendingExecAction = PendingExecAction::LoadCommit;
        appendExecUop(macroCmd, loadLatencyBase);
        return;
    }

    runtime.pendingExecAction = PendingExecAction::LoadRetry;
    appendExecUop(macroCmd, clockPeriod());
}

void
MpuUnit::appendFusedComputeProgressUop(MacroCmdContext &macroCmd,
                                       MpuMacroRuntime &runtime)
{
    const ParsedCmd &cmd = runtime.parsed;
    if (fusedComputeReady(cmd)) {
        runtime.pendingExecAction = PendingExecAction::LoadCommit;
        lastComputeLatencyCyclesValue = static_cast<uint64_t>(cmd.k);
        lastOutputReadyLatencyCyclesValue =
            static_cast<uint64_t>(cmd.k) + arrayDim;
        appendExecUop(macroCmd,
                      static_cast<Tick>(cmd.k) * clockPeriod());
        return;
    }

    runtime.pendingExecAction = PendingExecAction::LoadRetry;
    appendExecUop(macroCmd, clockPeriod());
}

uint32_t
MpuUnit::fusedMatmulTileCount(const ParsedCmd &cmd) const
{
    const uint32_t tile_rows = (cmd.m + cmd.tileM - 1) / cmd.tileM;
    const uint32_t tile_cols = (cmd.n + cmd.tileN - 1) / cmd.tileN;
    return tile_rows * tile_cols;
}

void
MpuUnit::beginFusedMatmulTile(MacroCmdContext &macroCmd,
                              MpuMacroRuntime &runtime)
{
    const ParsedCmd &cmd = runtime.parsed;
    const uint32_t tile_cols = (cmd.n + cmd.tileN - 1) / cmd.tileN;
    const uint32_t tile = runtime.fusedNextTile;
    const uint32_t tile_row = tile / tile_cols;
    const uint32_t tile_col = tile % tile_cols;
    const Addr a_tile_base = cmd.aBase +
        static_cast<Addr>(tile_row) * cmd.tileM * cmd.tileK;
    const Addr b_tile_base = cmd.bBase +
        static_cast<Addr>(tile_col) * cmd.tileK * cmd.tileN;
    runtime.fusedStage = FusedMatmulStage::Loading;
    runtime.fusedTileOrdinal = tile;
    runtime.fusedRow0 = (tile / tile_cols) * cmd.tileM;
    runtime.fusedCol0 = (tile % tile_cols) * cmd.tileN;
    runtime.fusedRows = std::min(cmd.tileM, cmd.m - runtime.fusedRow0);
    runtime.fusedCols = std::min(cmd.tileN, cmd.n - runtime.fusedCol0);
    runtime.fusedLoadResponses = 0;
    runtime.fusedAData.assign(static_cast<size_t>(runtime.fusedRows) * cmd.k,
                              0);
    runtime.fusedBData.assign(static_cast<size_t>(cmd.k) * runtime.fusedCols,
                              0);
    runtime.fusedCData.assign(static_cast<size_t>(runtime.fusedRows) *
                                  runtime.fusedCols,
                              0);

    if (!runtime.memWindow.active) {
        beginMemWindow(runtime);
    }

    appendLoadUop(macroCmd, a_tile_base, runtime.fusedAData.size());
    auto &a_uop = macroCmd.uopQueue.back();
    a_uop.portId = mvinAPortId();
    a_uop.token = makeFusedMatmulToken(FusedMatmulTokenKind::ALoad, 0);

    appendLoadUop(macroCmd, b_tile_base, runtime.fusedBData.size());
    auto &b_uop = macroCmd.uopQueue.back();
    b_uop.portId = mvinBPortId();
    b_uop.token = makeFusedMatmulToken(FusedMatmulTokenKind::BLoad, 0);
}

void
MpuUnit::performFusedMatmulTile(MpuMacroRuntime &runtime)
{
    const ParsedCmd &cmd = runtime.parsed;

    for (uint32_t row = 0; row < runtime.fusedRows; ++row) {
        for (uint32_t col = 0; col < runtime.fusedCols; ++col) {
            int32_t acc = 0;
            for (uint32_t depth = 0; depth < cmd.k; ++depth) {
                const int32_t a_val = static_cast<int32_t>(
                    runtime.fusedAData[row * cmd.k + depth]);
                const int32_t b_val = static_cast<int32_t>(
                    runtime.fusedBData[depth * runtime.fusedCols + col]);
                acc += a_val * b_val;
            }
            runtime.fusedCData[row * runtime.fusedCols + col] = acc;
        }
    }

    stats.totalMacOps += static_cast<uint64_t>(runtime.fusedRows) *
        runtime.fusedCols * cmd.k;
    stats.totalOutputElementsDrained +=
        static_cast<uint64_t>(runtime.fusedRows) * runtime.fusedCols;
}

void
MpuUnit::appendFusedMatmulStores(MacroCmdContext &macroCmd,
                                 MpuMacroRuntime &runtime)
{
    const ParsedCmd &cmd = runtime.parsed;
    const Addr c_tile_base = cmd.cBase +
        static_cast<Addr>(runtime.fusedTileOrdinal) *
            cmd.tileM * cmd.tileN * sizeof(int32_t);

    runtime.fusedStage = FusedMatmulStage::Storing;
    std::vector<uint8_t> bytes(
        runtime.fusedCData.size() * sizeof(int32_t), 0);
    std::memcpy(bytes.data(), runtime.fusedCData.data(), bytes.size());
    appendStoreUop(macroCmd, c_tile_base, bytes.size(), bytes);
    auto &uop = macroCmd.uopQueue.back();
    uop.portId = mvoutPortId();
    uop.token = makeFusedMatmulToken(FusedMatmulTokenKind::CStore, 0);
    runtime.fusedIssuedStores++;
}

MpuUnit::MacroCmdKind
MpuUnit::classifyMacroCmd(const std::vector<uint8_t> &cmd) const
{
    switch (parseCommand(cmd).kind) {
      case CmdKind::Mvin:
        return MacroCmdKind::Load;
      case CmdKind::Mvout:
        return MacroCmdKind::Store;
      case CmdKind::Load:
      case CmdKind::Compute:
      case CmdKind::ComputeFused:
      case CmdKind::FusedMatmul:
      case CmdKind::Drain:
        return MacroCmdKind::Exec;
    }

    panic("%s: unsupported MPU command kind in classifyMacroCmd", name());
}

uint32_t
MpuUnit::classifyIssueQueue(const std::vector<uint8_t> &cmd,
                            MacroCmdKind kind) const
{
    (void)kind;
    const CmdKind cmd_kind = parseCommand(cmd).kind;
    if (cmd_kind == CmdKind::Mvin) {
        return parseCommand(cmd).bufferKind == BufferKind::B ?
            PrefetchBIssueQueueId : PrefetchAIssueQueueId;
    }
    if (cmd_kind == CmdKind::Mvout) {
        return StoreIssueQueueId;
    }
    return ExecIssueQueueId;
}

std::vector<SpecializedExecutionUnit::IssueQueueState>
MpuUnit::buildIssueQueues() const
{
    return {
        {ExecIssueQueueId, IssueQueueKind::Exec, {}, {}, PortID(0)},
        {PrefetchAIssueQueueId, IssueQueueKind::Mem, {}, {}, mvinAPortId()},
        {PrefetchBIssueQueueId, IssueQueueKind::Mem, {}, {}, mvinBPortId()},
        {StoreIssueQueueId, IssueQueueKind::Mem, {}, {}, mvoutPortId()},
    };
}

bool
MpuUnit::canActivateMacroCmd(const MacroCmdContext &macroCmd) const
{
    const ParsedCmd cmd = parseCommand(macroCmd.cmd);
    if (cmd.kind == CmdKind::Mvout) {
        const CBufferSlot &slot = selectedCBuffer(cmd.bufferIndex);
        return slot.state == BufferState::Full &&
               slot.m == cmd.m &&
               slot.n == cmd.n;
    }

    if (cmd.kind == CmdKind::ComputeFused &&
        mpuFlagSet(cmd.flags, MpuFlagDrainToC)) {
        return selectedCBuffer(cmd.bufferIndex).state == BufferState::Empty;
    }

    return true;
}

void
MpuUnit::onMacroCmdBegin(MacroCmdContext &macroCmd)
{
    resetCommandStructures();

    MpuMacroRuntime runtime;
    runtime.parsed = parseCommand(macroCmd.cmd);
    validateCommand(macroCmd.cmd, runtime.parsed);
    runtime.commandStartTick = curTick();

    const ParsedCmd &cmd = runtime.parsed;
    pushQueueEntry(macroCmd.macroCmdId, cmd);
    macroRuntimes.emplace(macroCmd.macroCmdId, runtime);
    updateBusyAccounting(!macroRuntimes.empty());

    switch (cmd.kind) {
      case CmdKind::Mvin: {
        ABBufferSlot &slot = selectedABuffer(cmd.bufferKind, cmd.bufferIndex);
        slot.reset();
        slot.state = BufferState::LoadingFromSpm;
        slot.data.resize(requiredBytes(cmd), 0);
        break;
      }
      case CmdKind::Load:
        break;
      case CmdKind::Compute:
        outputStorage.reset();
        break;
      case CmdKind::ComputeFused:
      case CmdKind::FusedMatmul:
        break;
      case CmdKind::Drain:
        cBuffers[cmd.bufferIndex].reset();
        cBuffers[cmd.bufferIndex].state = BufferState::DrainingToBuffer;
        break;
      case CmdKind::Mvout:
        cBuffers[cmd.bufferIndex].state = BufferState::WritingToSpm;
        break;
    }

    refreshScoreboard();
    DPRINTF(MpuUnit,
            "begin kind=%u macro=%llu macro_fifo=%llu mem_q=%llu exec_q=%llu "
            "drain_q=%llu\n",
            static_cast<unsigned>(cmd.kind),
            static_cast<unsigned long long>(macroCmd.macroCmdId),
            static_cast<unsigned long long>(macroFifoOccupancy()),
            static_cast<unsigned long long>(memUopQueueOccupancy()),
            static_cast<unsigned long long>(execUopQueueOccupancy()),
            static_cast<unsigned long long>(drainUopQueueOccupancy()));
}

void
MpuUnit::buildUops(MacroCmdContext &macroCmd)
{
    auto &runtime = runtimeFor(macroCmd.macroCmdId);
    const ParsedCmd &cmd = runtime.parsed;

    switch (cmd.kind) {
      case CmdKind::Mvin:
        while (runtime.nextMemRow < expectedRows(cmd)) {
            appendMvinRowUop(macroCmd, runtime);
        }
        break;
      case CmdKind::Mvout:
        while (runtime.nextMemRow < cmd.m) {
            appendMvoutRowUop(macroCmd, runtime);
        }
        break;
      case CmdKind::Load:
        appendLoadProgressUop(macroCmd, runtime);
        break;
      case CmdKind::ComputeFused:
        appendFusedComputeProgressUop(macroCmd, runtime);
        break;
      case CmdKind::FusedMatmul:
        runtime.fusedNextTile = 0;
        runtime.fusedCompletedTiles = 0;
        runtime.fusedIssuedStores = 0;
        runtime.fusedStoreResponses = 0;
        beginFusedMatmulTile(macroCmd, runtime);
        break;
      case CmdKind::Compute:
        outputStorage.reset();
        outputStorage.state = OutputStorageState::Active;
        outputStorage.m = cmd.m;
        outputStorage.n = cmd.n;
        outputStorage.k = cmd.k;
        outputStorage.data.assign(static_cast<size_t>(cmd.m) * cmd.n, 0);
        lastComputeLatencyCyclesValue = static_cast<uint64_t>(cmd.k);
        lastOutputReadyLatencyCyclesValue =
            static_cast<uint64_t>(cmd.k) + arrayDim;
        outputStorage.readyTick =
            curTick() +
            static_cast<Tick>(lastOutputReadyLatencyCyclesValue) *
            clockPeriod();
        refreshScoreboard();
        DPRINTF(MpuUnit,
                "compute start dims=(%u,%u,%u) issue_cycles=%llu "
                "output_ready_cycles=%llu\n",
                cmd.m, cmd.n, cmd.k,
                static_cast<unsigned long long>(
                    lastComputeLatencyCyclesValue),
                static_cast<unsigned long long>(
                    lastOutputReadyLatencyCyclesValue));
        appendExecUop(macroCmd,
                      static_cast<Tick>(cmd.k) * clockPeriod());
        break;
      case CmdKind::Drain: {
        const Tick now = curTick();
        const Tick ready_tick = outputStorage.readyTick;
        const Tick wait_ticks = ready_tick > now ? ready_tick - now : 0;
        const Tick handoff_ticks = std::max(clockPeriod(), drainLatencyBase);
        const uint64_t wait_cycles =
            wait_ticks == 0 ? 0 : static_cast<uint64_t>(
                ticksToCycles(wait_ticks));

        lastDrainLatencyCyclesValue = wait_cycles +
            static_cast<uint64_t>(ticksToCycles(handoff_ticks));
        appendExecUop(macroCmd, wait_ticks + handoff_ticks);
        break;
      }
    }

    if (macroCmd.uopQueue.empty()) {
        markEpiloguePending(macroCmd);
    }
}

void
MpuUnit::onMemUopComplete(MacroCmdContext &macroCmd,
                          const MemTxnContext &txn, PacketPtr pkt)
{
    auto &runtime = runtimeFor(macroCmd.macroCmdId);
    const ParsedCmd &cmd = runtime.parsed;

    switch (cmd.kind) {
      case CmdKind::Mvin: {
        observeMemResponse(runtime);
        const uint8_t *src = pkt->getConstPtr<uint8_t>();
        ABBufferSlot &slot = selectedABuffer(cmd.bufferKind, cmd.bufferIndex);
        if (isContiguousSpmWindow(cmd)) {
            const size_t totalBytes = requiredBytes(cmd);
            panic_if(totalBytes > slot.data.size(),
                     "%s: contiguous mvin write exceeds destination buffer",
                     name());
            panic_if(pkt->getSize() != totalBytes,
                     "%s: contiguous mvin response size=%u expected=%zu",
                     name(), pkt->getSize(), totalBytes);
            std::memcpy(reinterpret_cast<uint8_t *>(slot.data.data()), src,
                        totalBytes);
            if (cmd.bufferKind == BufferKind::A) {
                stats.totalABytesIn += totalBytes;
            } else {
                stats.totalBBytesIn += totalBytes;
            }
            transitionABufferToFull(slot, cmd);
            finalizeMemWindow(runtime);
            refreshScoreboard();
            markEpiloguePending(macroCmd);
        } else {
            const uint32_t row = static_cast<uint32_t>(txn.token);
            const uint32_t rowBytes = expectedRowBytes(cmd);
            const size_t offset = static_cast<size_t>(row) * rowBytes;
            panic_if(offset + rowBytes > slot.data.size(),
                     "%s: mvin row write overflows destination buffer",
                     name());
            std::memcpy(reinterpret_cast<uint8_t *>(slot.data.data()) + offset,
                        src, rowBytes);

            if (cmd.bufferKind == BufferKind::A) {
                stats.totalABytesIn += rowBytes;
            } else {
                stats.totalBBytesIn += rowBytes;
            }

            DPRINTF(MpuUnit,
                    "mvin response row=%u addr=%#llx size=%u "
                    "buffer=%u idx=%u\n",
                    row, static_cast<unsigned long long>(txn.addr), txn.size,
                    static_cast<unsigned>(cmd.bufferKind), cmd.bufferIndex);

            if (runtime.nextMemRow >= expectedRows(cmd) &&
                macroCmd.outstandingMemUops == 0) {
                transitionABufferToFull(slot, cmd);
                finalizeMemWindow(runtime);
                refreshScoreboard();
                markEpiloguePending(macroCmd);
            }
        }
        break;
      }
      case CmdKind::Mvout:
        observeMemResponse(runtime);
        stats.totalCBytesOut += txn.size;
        if (isContiguousSpmWindow(cmd)) {
            finalizeMemWindow(runtime);
            markEpiloguePending(macroCmd);
        } else {
            DPRINTF(MpuUnit,
                    "mvout response row=%llu addr=%#llx size=%u idx=%u\n",
                    static_cast<unsigned long long>(txn.token),
                    static_cast<unsigned long long>(txn.addr), txn.size,
                    cmd.bufferIndex);
            if (runtime.nextMemRow >= cmd.m &&
                macroCmd.outstandingMemUops == 0) {
                finalizeMemWindow(runtime);
                markEpiloguePending(macroCmd);
            }
        }
        break;
      case CmdKind::FusedMatmul:
        observeMemResponse(runtime);
        if (txn.kind == MemTxnContext::Kind::Load) {
            const uint8_t *src = pkt->getConstPtr<uint8_t>();
            const FusedMatmulTokenKind token_kind =
                fusedMatmulTokenKind(txn.token);
            const uint32_t row = fusedMatmulTokenRow(txn.token);

            if (token_kind == FusedMatmulTokenKind::ALoad) {
                panic_if(row != 0 || txn.size != runtime.fusedAData.size(),
                         "%s: invalid fused A response row=%u size=%zu",
                         name(), row, txn.size);
                std::memcpy(runtime.fusedAData.data(), src,
                            runtime.fusedAData.size());
                stats.totalABytesIn += txn.size;
            } else if (token_kind == FusedMatmulTokenKind::BLoad) {
                panic_if(row != 0 || txn.size != runtime.fusedBData.size(),
                         "%s: invalid fused B response row=%u size=%zu",
                         name(), row, txn.size);
                std::memcpy(runtime.fusedBData.data(), src,
                            runtime.fusedBData.size());
                stats.totalBBytesIn += txn.size;
            } else {
                panic("%s: unexpected fused matmul load token kind", name());
            }

            runtime.fusedLoadResponses++;
            if (runtime.fusedLoadResponses == 2) {
                runtime.fusedStage = FusedMatmulStage::Computing;
                lastComputeLatencyCyclesValue = static_cast<uint64_t>(cmd.k);
                lastOutputReadyLatencyCyclesValue =
                    lastComputeLatencyCyclesValue + arrayDim;
                performFusedMatmulTile(runtime);
                appendExecUop(macroCmd,
                              static_cast<Tick>(lastComputeLatencyCyclesValue) *
                                  clockPeriod());
                appendFusedMatmulStores(macroCmd, runtime);
                runtime.fusedCompletedTiles++;
                runtime.fusedNextTile++;
                if (runtime.fusedNextTile < fusedMatmulTileCount(cmd)) {
                    beginFusedMatmulTile(macroCmd, runtime);
                }
            }
        } else {
            panic_if(fusedMatmulTokenKind(txn.token) !=
                         FusedMatmulTokenKind::CStore,
                     "%s: unexpected fused matmul store token kind", name());
            stats.totalCBytesOut += txn.size;
            runtime.fusedStoreResponses++;
            if (runtime.fusedNextTile >= fusedMatmulTileCount(cmd) &&
                runtime.fusedStoreResponses == runtime.fusedIssuedStores &&
                macroCmd.outstandingMemUops == 0) {
                runtime.fusedStage = FusedMatmulStage::Done;
                finalizeMemWindow(runtime);
                markEpiloguePending(macroCmd);
            }
        }
        break;
      case CmdKind::Load:
      case CmdKind::Compute:
      case CmdKind::ComputeFused:
      case CmdKind::Drain:
        break;
    }
}

void
MpuUnit::onExecUopComplete(MacroCmdContext &macroCmd,
                           const MicroOpContext &uop)
{
    auto &runtime = runtimeFor(macroCmd.macroCmdId);
    const ParsedCmd &cmd = runtime.parsed;

    if (uop.kind != MicroOpContext::Kind::Exec) {
        return;
    }

    switch (cmd.kind) {
      case CmdKind::Load: {
        if (runtime.pendingExecAction == PendingExecAction::LoadRetry) {
            stats.stallCyclesBufferHazard++;
            appendLoadProgressUop(macroCmd, runtime);
            return;
        }
        panic_if(runtime.pendingExecAction != PendingExecAction::LoadCommit,
                 "%s: load macro %llu completed without a committed load uop",
                 name(),
                 static_cast<unsigned long long>(macroCmd.macroCmdId));
        runtime.pendingExecAction = PendingExecAction::None;
        ABBufferSlot &slot = selectedABuffer(cmd.bufferKind, cmd.bufferIndex);
        transitionABufferToLoaded(slot, cmd);
        break;
      }
      case CmdKind::Compute:
        performCompute(cmd);
        break;
      case CmdKind::ComputeFused:
        if (runtime.pendingExecAction == PendingExecAction::LoadRetry) {
            stats.stallCyclesBufferHazard++;
            appendFusedComputeProgressUop(macroCmd, runtime);
            return;
        }
        panic_if(runtime.pendingExecAction != PendingExecAction::LoadCommit,
                 "%s: fused compute macro %llu completed without a committed "
                 "uop",
                 name(),
                 static_cast<unsigned long long>(macroCmd.macroCmdId));
        runtime.pendingExecAction = PendingExecAction::None;
        performComputeFused(cmd);
        break;
      case CmdKind::FusedMatmul:
        panic_if(runtime.fusedStage == FusedMatmulStage::Idle,
                 "%s: fused matmul exec completed before tile start", name());
        refreshScoreboard();
        DPRINTF(MpuUnit,
                "fused matmul tile=%u row0=%u col0=%u rows=%u cols=%u\n",
                runtime.fusedTileOrdinal, runtime.fusedRow0,
                runtime.fusedCol0, runtime.fusedRows, runtime.fusedCols);
        break;
      case CmdKind::Drain:
        performDrain(cmd);
        break;
      case CmdKind::Mvin:
      case CmdKind::Mvout:
        break;
    }

    refreshScoreboard();
    DPRINTF(MpuUnit,
            "uop complete kind=%u a=(%d,%d) b=(%d,%d) out=%d\n",
            static_cast<unsigned>(cmd.kind), aBufferState(0), aBufferState(1),
            bBufferState(0), bBufferState(1), outputStorageStateCode());
    markEpiloguePending(macroCmd);
}

void
MpuUnit::onMacroCmdEnd(MacroCmdContext &macroCmd)
{
    const auto it = macroRuntimes.find(macroCmd.macroCmdId);
    panic_if(it == macroRuntimes.end(),
             "%s: missing MPU runtime during epilogue for macro %llu",
             name(), static_cast<unsigned long long>(macroCmd.macroCmdId));
    const ParsedCmd &cmd = it->second.parsed;
    switch (cmd.kind) {
      case CmdKind::Mvin:
        stats.mvinCmdCount++;
        break;
      case CmdKind::Load:
        stats.loadCmdCount++;
        break;
      case CmdKind::Compute:
      case CmdKind::ComputeFused:
      case CmdKind::FusedMatmul:
        stats.computeCmdCount++;
        break;
      case CmdKind::Drain:
        stats.drainCmdCount++;
        break;
      case CmdKind::Mvout:
        finalizeMemWindow(macroRuntimes.at(macroCmd.macroCmdId));
        cBuffers[cmd.bufferIndex].reset();
        stats.mvoutCmdCount++;
        break;
    }

    lastCommandLatencyCyclesValue =
        elapsedCyclesSince(it->second.commandStartTick);
    popQueueEntry(macroCmd.macroCmdId, cmd);
    macroRuntimes.erase(it);
    refreshScoreboard();
    updateBusyAccounting(!macroRuntimes.empty());

    DPRINTF(MpuUnit,
            "epilogue complete macro_fifo=%llu mem_q=%llu exec_q=%llu "
            "drain_q=%llu cmd_latency_cycles=%llu\n",
            static_cast<unsigned long long>(macroFifoOccupancy()),
            static_cast<unsigned long long>(memUopQueueOccupancy()),
            static_cast<unsigned long long>(execUopQueueOccupancy()),
            static_cast<unsigned long long>(drainUopQueueOccupancy()),
            static_cast<unsigned long long>(lastCommandLatencyCyclesValue));
}

uint64_t
MpuUnit::macroFifoOccupancy() const
{
    return queueOccupancy() + macroCmdFifo.size();
}

uint64_t
MpuUnit::memUopQueueOccupancy() const
{
    return memUopQueue.size();
}

uint64_t
MpuUnit::execUopQueueOccupancy() const
{
    return execUopQueue.size();
}

uint64_t
MpuUnit::drainUopQueueOccupancy() const
{
    return drainUopQueue.size();
}

uint64_t
MpuUnit::mvinCmdCount() const
{
    return stats.mvinCmdCount.value();
}

uint64_t
MpuUnit::loadCmdCount() const
{
    return stats.loadCmdCount.value();
}

uint64_t
MpuUnit::computeCmdCount() const
{
    return stats.computeCmdCount.value();
}

uint64_t
MpuUnit::drainCmdCount() const
{
    return stats.drainCmdCount.value();
}

uint64_t
MpuUnit::mvoutCmdCount() const
{
    return stats.mvoutCmdCount.value();
}

uint64_t
MpuUnit::totalABytesIn() const
{
    return stats.totalABytesIn.value();
}

uint64_t
MpuUnit::totalBBytesIn() const
{
    return stats.totalBBytesIn.value();
}

uint64_t
MpuUnit::totalCBytesOut() const
{
    return stats.totalCBytesOut.value();
}

uint64_t
MpuUnit::totalOutputElementsDrained() const
{
    return stats.totalOutputElementsDrained.value();
}

uint64_t
MpuUnit::totalMacOps() const
{
    return stats.totalMacOps.value();
}

uint64_t
MpuUnit::busyCycles() const
{
    return currentBusyCycles();
}

uint64_t
MpuUnit::idleCycles() const
{
    return currentIdleCycles();
}

uint64_t
MpuUnit::stallCyclesWaitingForSpm() const
{
    return stats.stallCyclesWaitingForSpm.value();
}

uint64_t
MpuUnit::stallCyclesBufferHazard() const
{
    return stats.stallCyclesBufferHazard.value();
}

uint64_t
MpuUnit::stallCyclesOutputStorageUnavailable() const
{
    return stats.stallCyclesOutputStorageUnavailable.value();
}

uint64_t
MpuUnit::stallCyclesDrainDestBusy() const
{
    return stats.stallCyclesDrainDestBusy.value();
}

uint64_t
MpuUnit::lastComputeLatencyCycles() const
{
    return lastComputeLatencyCyclesValue;
}

uint64_t
MpuUnit::lastDrainLatencyCycles() const
{
    return lastDrainLatencyCyclesValue;
}

uint64_t
MpuUnit::lastOutputReadyLatencyCycles() const
{
    return lastOutputReadyLatencyCyclesValue;
}

uint64_t
MpuUnit::lastCommandLatencyCycles() const
{
    return lastCommandLatencyCyclesValue;
}

int
MpuUnit::currentCmdKind() const
{
    if (macroRuntimes.empty()) {
        return -1;
    }
    return static_cast<int>(macroRuntimes.begin()->second.parsed.kind);
}

int
MpuUnit::aBufferState(uint32_t index) const
{
    panic_if(index > 1, "%s: invalid A buffer index %u", name(), index);
    return static_cast<int>(aBuffers[index].state);
}

int
MpuUnit::bBufferState(uint32_t index) const
{
    panic_if(index > 1, "%s: invalid B buffer index %u", name(), index);
    return static_cast<int>(bBuffers[index].state);
}

int
MpuUnit::cBufferState(uint32_t index) const
{
    panic_if(index > 1, "%s: invalid C buffer index %u", name(), index);
    return static_cast<int>(cBuffers[index].state);
}

int
MpuUnit::outputStorageStateCode() const
{
    return static_cast<int>(outputStorage.state);
}

int
MpuUnit::loadedAIndex() const
{
    return loadedA.valid ? static_cast<int>(loadedA.bufferIndex) : -1;
}

int
MpuUnit::loadedBIndex() const
{
    return loadedB.valid ? static_cast<int>(loadedB.bufferIndex) : -1;
}

bool
MpuUnit::scoreboardLoadedAReady() const
{
    return scoreboard.loadedAReady;
}

bool
MpuUnit::scoreboardLoadedBReady() const
{
    return scoreboard.loadedBReady;
}

bool
MpuUnit::scoreboardOutputReady() const
{
    return scoreboard.outputStorageReady;
}

} // namespace gem5
