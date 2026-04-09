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
      drainLatencyBase(params.drain_latency_base)
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
    fatal_if(memSidePorts.empty() || memSidePorts.size() > 2,
             "%s: MpuUnit requires one or two mem_side ports", name());

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
MpuUnit::validateLoad(const ParsedCmd &cmd) const
{
    panic_if(cmd.bufferKind == BufferKind::C,
             "%s: load(C) is not supported in MPU v1.1", name());
    panic_if(cmd.bufferKind == BufferKind::Reserved,
             "%s: load requires an A or B source buffer", name());

    const ABBufferSlot &slot = selectedABuffer(cmd.bufferKind,
                                               cmd.bufferIndex);
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
    panic_if(cmd.m > arrayDim || cmd.n > arrayDim,
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
    panic_if(cmd.flags != 0 || extractWord(rawCmd, ReservedWord13) != 0 ||
                 extractWord(rawCmd, ReservedWord14) != 0 ||
                 extractWord(rawCmd, ReservedWord15) != 0,
             "%s: MPU command reserved words must be zero", name());

    switch (cmd.kind) {
      case CmdKind::Mvin:
        validateMvin(cmd);
        return;
      case CmdKind::Mvout:
        validateMvout(cmd);
        return;
      case CmdKind::Load:
        validateLoad(cmd);
        return;
      case CmdKind::Compute:
        validateCompute(cmd);
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
    panic_if(!memUopQueue.empty() || !execUopQueue.empty() ||
                 !drainUopQueue.empty(),
             "%s: per-command uop queues must be empty at command start",
             name());
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
MpuUnit::mvinPortId() const
{
    return 0;
}

PortID
MpuUnit::mvoutPortId() const
{
    return memSidePorts.size() > 1 ? 1 : 0;
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

void
MpuUnit::pushQueueEntry(const ParsedCmd &cmd)
{
    QueueEntry entry{cmd.kind, cmd.bufferKind, cmd.bufferIndex,
                     cmd.m, cmd.n, cmd.k};

    macroCmdFifo.push_back(entry);
    panic_if(macroCmdFifo.size() > cmdQueueDepth,
             "%s: macro FIFO overflow", name());

    switch (cmd.kind) {
      case CmdKind::Mvin:
      case CmdKind::Mvout:
        panic_if(memUopQueue.size() >= memUopQueueDepth,
                 "%s: mem uop queue overflow", name());
        memUopQueue.push_back(entry);
        break;
      case CmdKind::Load:
      case CmdKind::Compute:
        panic_if(execUopQueue.size() >= execUopQueueDepth,
                 "%s: exec uop queue overflow", name());
        execUopQueue.push_back(entry);
        break;
      case CmdKind::Drain:
        panic_if(drainUopQueue.size() >= drainUopQueueDepth,
                 "%s: drain uop queue overflow", name());
        drainUopQueue.push_back(entry);
        break;
    }
}

void
MpuUnit::popQueueEntry(const ParsedCmd &cmd)
{
    switch (cmd.kind) {
      case CmdKind::Mvin:
      case CmdKind::Mvout:
        if (!memUopQueue.empty()) {
            memUopQueue.pop_front();
        }
        break;
      case CmdKind::Load:
      case CmdKind::Compute:
        if (!execUopQueue.empty()) {
            execUopQueue.pop_front();
        }
        break;
      case CmdKind::Drain:
        if (!drainUopQueue.empty()) {
            drainUopQueue.pop_front();
        }
        break;
    }

    if (!macroCmdFifo.empty()) {
        macroCmdFifo.pop_front();
    }
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

    appendLoadUop(macroCmd,
                  cmd.spmAddr + static_cast<Addr>(row) * cmd.strideBytes,
                  expectedRowBytes(cmd));
    macroCmd.uopQueue.back().portId = mvinPortId();
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

    appendStoreUop(macroCmd,
                   cmd.spmAddr + static_cast<Addr>(row) * cmd.strideBytes,
                   cmd.n * sizeof(int32_t),
                   serializeCRow(selectedCBuffer(cmd.bufferIndex), row));
    macroCmd.uopQueue.back().portId = mvoutPortId();
    macroCmd.uopQueue.back().token = row;
    runtime.nextMemRow++;
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
      case CmdKind::Drain:
        return MacroCmdKind::Exec;
    }

    panic("%s: unsupported MPU command kind in classifyMacroCmd", name());
}

uint32_t
MpuUnit::classifyIssueQueue(const std::vector<uint8_t> &cmd,
                            MacroCmdKind kind) const
{
    (void)cmd;
    (void)kind;
    return 0;
}

std::vector<SpecializedExecutionUnit::IssueQueueState>
MpuUnit::buildIssueQueues() const
{
    return {{0, IssueQueueKind::Exec, {}, {}, PortID(0)}};
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
    pushQueueEntry(cmd);
    macroRuntimes.emplace(macroCmd.macroCmdId, runtime);
    updateBusyAccounting(true);

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
        appendMvinRowUop(macroCmd, runtime);
        break;
      case CmdKind::Mvout:
        appendMvoutRowUop(macroCmd, runtime);
        break;
      case CmdKind::Load:
        appendExecUop(macroCmd, loadLatencyBase);
        break;
      case CmdKind::Compute:
        outputStorage.reset();
        outputStorage.state = OutputStorageState::Active;
        outputStorage.m = cmd.m;
        outputStorage.n = cmd.n;
        outputStorage.k = cmd.k;
        outputStorage.data.assign(static_cast<size_t>(cmd.m) * cmd.n, 0);
        lastComputeLatencyCyclesValue =
            static_cast<uint64_t>(cmd.k) + cmd.m;
        refreshScoreboard();
        DPRINTF(MpuUnit,
                "compute start dims=(%u,%u,%u) latency_cycles=%llu\n",
                cmd.m, cmd.n, cmd.k,
                static_cast<unsigned long long>(
                    lastComputeLatencyCyclesValue));
        appendExecUop(macroCmd,
                      (static_cast<Tick>(cmd.k) + cmd.m) * clockPeriod());
        break;
      case CmdKind::Drain:
        appendExecUop(macroCmd,
                      drainLatencyBase +
                      static_cast<Tick>(cmd.m) * cmd.n * clockPeriod());
        break;
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
        const uint32_t row = static_cast<uint32_t>(txn.token);
        const uint32_t rowBytes = expectedRowBytes(cmd);
        const uint8_t *src = pkt->getConstPtr<uint8_t>();
        ABBufferSlot &slot = selectedABuffer(cmd.bufferKind, cmd.bufferIndex);
        const size_t offset = static_cast<size_t>(row) * rowBytes;
        panic_if(offset + rowBytes > slot.data.size(),
                 "%s: mvin row write overflows destination buffer", name());
        std::memcpy(reinterpret_cast<uint8_t *>(slot.data.data()) + offset,
                    src, rowBytes);

        if (cmd.bufferKind == BufferKind::A) {
            stats.totalABytesIn += rowBytes;
        } else {
            stats.totalBBytesIn += rowBytes;
        }

        DPRINTF(MpuUnit,
                "mvin response row=%u addr=%#llx size=%u buffer=%u idx=%u\n",
                row, static_cast<unsigned long long>(txn.addr), txn.size,
                static_cast<unsigned>(cmd.bufferKind), cmd.bufferIndex);

        if (runtime.nextMemRow < expectedRows(cmd)) {
            appendMvinRowUop(macroCmd, runtime);
        } else {
            transitionABufferToFull(slot, cmd);
            finalizeMemWindow(runtime);
            refreshScoreboard();
            markEpiloguePending(macroCmd);
        }
        break;
      }
      case CmdKind::Mvout:
        observeMemResponse(runtime);
        stats.totalCBytesOut += txn.size;
        DPRINTF(MpuUnit,
                "mvout response row=%llu addr=%#llx size=%u idx=%u\n",
                static_cast<unsigned long long>(txn.token),
                static_cast<unsigned long long>(txn.addr), txn.size,
                cmd.bufferIndex);
        if (runtime.nextMemRow < cmd.m) {
            appendMvoutRowUop(macroCmd, runtime);
        } else {
            finalizeMemWindow(runtime);
            markEpiloguePending(macroCmd);
        }
        break;
      case CmdKind::Load:
      case CmdKind::Compute:
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
        ABBufferSlot &slot = selectedABuffer(cmd.bufferKind, cmd.bufferIndex);
        transitionABufferToLoaded(slot, cmd);
        break;
      }
      case CmdKind::Compute:
        performCompute(cmd);
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
    popQueueEntry(cmd);
    macroRuntimes.erase(it);
    refreshScoreboard();
    updateBusyAccounting(false);

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
