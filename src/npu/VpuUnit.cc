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

#include "npu/VpuUnit.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/VPU.hh"

namespace gem5
{

namespace
{

constexpr uint32_t LocalRegionTagMask = 0xFF000000U;
constexpr uint32_t LocalRegionInputTag = 0x80000000U;
constexpr uint32_t LocalRegionOutputTag = 0x81000000U;

} // anonymous namespace

VpuUnit::VpuUnit(const VpuUnitParams &params)
    : SpecializedExecutionUnit(params), deviceId(params.device_id),
      lut(params.lut),
      numMemPorts(params.num_mem_side_ports),
      numInputPorts(params.num_input_ports),
      numOutputPorts(params.num_output_ports),
      inputBufferCount(params.input_buffer_count),
      outputBufferCount(params.output_buffer_count),
      localInputBase(params.local_input_base),
      localOutputBase(params.local_output_base),
      localBufferStride(params.local_buffer_stride),
      inputBuffers(numInputPorts,
                   std::vector<LocalBufferSlot>(inputBufferCount)),
      outputBuffers(numOutputPorts,
                    std::vector<LocalBufferSlot>(outputBufferCount))
{
    panic_if(lut == nullptr, "%s: lut must not be null", name());
    panic_if(numMemPorts == 0, "%s: mem ports must be non-zero", name());
    panic_if(numInputPorts == 0 || numOutputPorts == 0,
             "%s: input/output ports must be non-zero", name());
    panic_if(inputBufferCount == 0 || outputBufferCount == 0,
             "%s: input/output buffer counts must be non-zero", name());
    panic_if(
        localBufferStride == 0,
        "%s: local buffer stride must be non-zero", name());
}

uint32_t
VpuUnit::unpackWord(const std::vector<uint8_t> &bytes) const
{
    uint32_t value = 0;
    if (!bytes.empty()) {
        std::memcpy(&value, bytes.data(),
                    std::min(bytes.size(), sizeof(value)));
    }
    return value;
}

std::vector<uint8_t>
VpuUnit::packWord(uint32_t value) const
{
    std::vector<uint8_t> bytes(sizeof(value), 0);
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

uint32_t
VpuUnit::cmdWord(const std::vector<uint8_t> &cmd, size_t wordIdx) const
{
    return readCmdWord(cmd, wordIdx);
}

Addr
VpuUnit::slotAddr(PortID portId, Addr offset) const
{
    return SpmBase + (static_cast<Addr>(portId) * SpmSlotStride) + offset;
}

VpuUnit::Opcode
VpuUnit::decodeOpcode(uint8_t opCode) const
{
    switch (opCode) {
      case static_cast<uint8_t>(Opcode::Exec):
        return Opcode::Exec;
      case static_cast<uint8_t>(Opcode::VAdd):
        return Opcode::VAdd;
      case static_cast<uint8_t>(Opcode::VSub):
        return Opcode::VSub;
      case static_cast<uint8_t>(Opcode::VMul):
        return Opcode::VMul;
      case static_cast<uint8_t>(Opcode::VDiv):
        return Opcode::VDiv;
      case static_cast<uint8_t>(Opcode::VScale):
        return Opcode::VScale;
      case static_cast<uint8_t>(Opcode::VCvtI2F):
        return Opcode::VCvtI2F;
      case static_cast<uint8_t>(Opcode::VCvtF2I):
        return Opcode::VCvtF2I;
      case static_cast<uint8_t>(Opcode::VSqrt):
        return Opcode::VSqrt;
      case static_cast<uint8_t>(Opcode::VFma):
        return Opcode::VFma;
      case static_cast<uint8_t>(Opcode::VReduceSum):
        return Opcode::VReduceSum;
      case static_cast<uint8_t>(Opcode::VReduceMax):
        return Opcode::VReduceMax;
      case static_cast<uint8_t>(Opcode::VLoad):
        return Opcode::VLoad;
      case static_cast<uint8_t>(Opcode::VStore):
        return Opcode::VStore;
      case static_cast<uint8_t>(Opcode::VExp):
        return Opcode::VExp;
      case static_cast<uint8_t>(Opcode::VSoftmax):
        return Opcode::VSoftmax;
      default:
        fatal("%s: unsupported VPU opCode=%u", name(), opCode);
    }
}

VpuUnit::DecodedVectorOp
VpuUnit::decodeVectorOp(const MacroCmdContext &macroCmd) const
{
    DecodedVectorOp op;
    op.opcode = decodeOpcode(macroCmd.fields.opCode);
    op.legacyExec = op.opcode == Opcode::Exec;
    op.flags = cmdWord(macroCmd.cmd, FlagsWord);
    op.elemCount = cmdWord(macroCmd.cmd, ElemCountWord);
    op.srcStrideBytes = cmdWord(macroCmd.cmd, SrcStrideWord);
    op.dstStrideBytes = cmdWord(macroCmd.cmd, DstStrideWord);
    op.scalarBits = cmdWord(macroCmd.cmd, ScalarBitsWord);
    op.repetition = std::max<uint32_t>(
        1, cmdWord(macroCmd.cmd, RepetitionWord));

    const uint32_t rawDataType = cmdWord(macroCmd.cmd, DataTypeWord);
    switch (rawDataType) {
      case static_cast<uint32_t>(DataType::Int32):
        op.dataType = DataType::Int32;
        break;
      case static_cast<uint32_t>(DataType::Float32):
        op.dataType = DataType::Float32;
        break;
      default:
        fatal("%s: unsupported VPU dataType=%u", name(), rawDataType);
    }

    if (op.elemCount == 0) {
        op.elemCount = 1;
    }
    if (op.srcStrideBytes == 0) {
        op.srcStrideBytes = sizeof(uint32_t);
    }
    if (op.dstStrideBytes == 0) {
        op.dstStrideBytes = sizeof(uint32_t);
    }

    op.elemSize = sizeof(uint32_t);
    op.srcSpanBytes =
        (static_cast<size_t>(op.elemCount - 1) * op.srcStrideBytes) +
        op.elemSize;
    op.dstSpanBytes =
        (static_cast<size_t>(op.elemCount - 1) * op.dstStrideBytes) +
        op.elemSize;

    return op;
}

std::vector<PortID>
VpuUnit::decodeMask(uint32_t mask) const
{
    std::vector<PortID> ports;
    for (PortID port = 0; port < static_cast<PortID>(numMemPorts); ++port) {
        if ((mask & (1U << port)) != 0) {
            ports.push_back(port);
        }
    }
    return ports;
}

void
VpuUnit::validatePortLayout(const VpuMacroState &state) const
{
    const auto &op = state.op;
    if (op.opcode == Opcode::VLoad) {
        panic_if(
            state.readPorts.size() != 1,
            "%s: VLOAD expects one source port per macro command", name());
        return;
    }
    if (op.opcode == Opcode::VStore) {
        panic_if(state.writePorts.size() != 1,
                 "%s: VSTORE expects one destination port per macro command",
                 name());
        return;
    }

    switch (op.opcode) {
      case Opcode::Exec:
        panic_if(state.writePorts.empty(),
                 "%s: legacy exec requires write ports", name());
        break;
      case Opcode::VAdd:
      case Opcode::VSub:
      case Opcode::VMul:
      case Opcode::VDiv:
        panic_if(state.writePorts.empty() ||
                 state.readPorts.size() != state.writePorts.size() * 2,
                 "%s: binary op requires exactly two reads per write",
                 name());
        break;
      case Opcode::VFma:
        panic_if(state.writePorts.empty() ||
                 state.readPorts.size() != state.writePorts.size() * 3,
                 "%s: VFMA requires exactly three reads per write", name());
        break;
      case Opcode::VReduceSum:
      case Opcode::VReduceMax:
      case Opcode::VScale:
      case Opcode::VCvtI2F:
      case Opcode::VCvtF2I:
      case Opcode::VSqrt:
      case Opcode::VExp:
      case Opcode::VSoftmax:
        panic_if(state.writePorts.empty() ||
                 state.readPorts.size() != state.writePorts.size(),
                 "%s: unary/reduce op requires one read per write", name());
        break;
      case Opcode::VLoad:
      case Opcode::VStore:
        break;
    }
}

bool
VpuUnit::isSpmAddr(Addr addr) const
{
    return addr >= SpmBase && addr < (SpmBase + (numMemPorts * SpmSlotStride));
}

bool
VpuUnit::isInputLocalAddr(Addr addr) const
{
    return (addr & LocalRegionTagMask) ==
           (localInputBase & LocalRegionTagMask);
}

bool
VpuUnit::isOutputLocalAddr(Addr addr) const
{
    return (addr & LocalRegionTagMask) ==
           (localOutputBase & LocalRegionTagMask);
}

VpuUnit::LocalAddr
VpuUnit::decodeLocalAddr(
    Addr addr, BufferRole expected, size_t accessSize) const
{
    const Addr base = expected == BufferRole::Input ? localInputBase :
        localOutputBase;
    const uint32_t baseTag = static_cast<uint32_t>(base & LocalRegionTagMask);
    panic_if((addr & LocalRegionTagMask) != baseTag,
             "%s: address %#llx is not in expected local-buffer region",
             name(), static_cast<unsigned long long>(addr));

    const uint32_t bufferCount =
        expected == BufferRole::Input ? inputBufferCount : outputBufferCount;
    const Addr offset = addr - base;
    const uint32_t slotIndex = offset / localBufferStride;
    const size_t intraOffset = offset % localBufferStride;

    panic_if(intraOffset + accessSize > localBufferStride,
             "%s: local-buffer access exceeds slot stride", name());

    panic_if(slotIndex >= bufferCount,
             "%s: local-buffer index %u out of range", name(), slotIndex);
    return {expected, 0, slotIndex, intraOffset};
}

PortID
VpuUnit::decodeSpmPort(Addr addr, size_t accessSize) const
{
    panic_if(!isSpmAddr(addr), "%s: address %#llx is not in SPM range", name(),
             static_cast<unsigned long long>(addr));
    const Addr offset = addr - SpmBase;
    const PortID port = offset / SpmSlotStride;
    const size_t intra = offset % SpmSlotStride;
    panic_if(intra + accessSize > SpmSlotStride,
             "%s: SPM access exceeds slot stride", name());
    return port;
}

void
VpuUnit::validateCommand(const MacroCmdContext &macroCmd,
                         VpuMacroState &state) const
{
    fatal_if(macroCmd.fields.deviceType != VpuDeviceType,
             "%s: unexpected deviceType=%u for VPU command", name(),
             macroCmd.fields.deviceType);
    fatal_if(macroCmd.fields.deviceId != deviceId,
             "%s: command deviceId=%u does not match instance deviceId=%u",
             name(), macroCmd.fields.deviceId, deviceId);

    state.readMask = cmdWord(macroCmd.cmd, ReadMaskWord);
    state.writeMask = cmdWord(macroCmd.cmd, WriteMaskWord);
    state.readPorts = decodeMask(state.readMask);
    state.writePorts = decodeMask(state.writeMask);
    state.src0Addr = cmdWord(macroCmd.cmd, Src0AddrWord);
    state.src1Addr = cmdWord(macroCmd.cmd, Src1AddrWord);
    state.src2Addr = cmdWord(macroCmd.cmd, Src2AddrWord);
    state.dstAddr = cmdWord(macroCmd.cmd, DstAddrWord);

    validatePortLayout(state);

    if (state.op.opcode == Opcode::VLoad) {
        panic_if(!isSpmAddr(state.src0Addr),
                 "%s: VLOAD source must be in SPM", name());
        panic_if(!isInputLocalAddr(state.dstAddr),
                 "%s: VLOAD destination must be input local-buffer", name());
    } else if (state.op.opcode == Opcode::VStore) {
        panic_if(!isInputLocalAddr(state.src0Addr) &&
                     !isOutputLocalAddr(state.src0Addr),
                 "%s: VSTORE source must be a local-buffer address", name());
        panic_if(!isSpmAddr(state.dstAddr),
                 "%s: VSTORE destination must be in SPM", name());
    } else {
        panic_if(!isInputLocalAddr(state.src0Addr),
                 "%s: compute src0 must be input local-buffer", name());
        if (!state.readPorts.empty()) {
            panic_if(!isInputLocalAddr(state.src1Addr) &&
                         state.readPorts.size() > 1,
                     "%s: compute src1 must be input local-buffer", name());
        }
        if (state.op.opcode == Opcode::VFma) {
            panic_if(!isInputLocalAddr(state.src2Addr),
                     "%s: compute src2 must be input local-buffer", name());
        }
        panic_if(
            !isOutputLocalAddr(state.dstAddr),
            "%s: compute destination must be output local-buffer", name());
    }

    if (state.op.opcode == Opcode::VCvtI2F) {
        panic_if(state.op.dataType != DataType::Float32,
                 "%s: VCvtI2F requires Float32 destination", name());
    }
    if (state.op.opcode == Opcode::VCvtF2I) {
        panic_if(state.op.dataType != DataType::Int32,
                 "%s: VCvtF2I requires Int32 destination", name());
    }
    if ((state.op.opcode == Opcode::VSqrt ||
         state.op.opcode == Opcode::VExp ||
         state.op.opcode == Opcode::VSoftmax ||
         state.op.opcode == Opcode::VFma) &&
        state.op.dataType != DataType::Float32) {
        panic_if(true, "%s: floating-point VPU op requires Float32", name());
    }
}

Tick
VpuUnit::computeExecLatency(const VpuMacroState &state)
{
    Tick extraLatency = 0;
    if (isLutOpcode(state.op.opcode)) {
        const uint32_t lutRequests =
            state.op.opcode == Opcode::VSoftmax ?
            (state.op.elemCount * state.writePorts.size()) :
            (state.op.elemCount * state.writePorts.size());
        extraLatency = lut->reserve(lutOperation(state.op.opcode), lutRequests,
                                    curTick());
    }

    const Tick baseLatency = debugProcessLatency;
    const Tick totalLatency = baseLatency + extraLatency;
    if (isLinearOpcode(state.op.opcode)) {
        lastLinearExecuteLatencyValue = totalLatency;
    }
    return totalLatency;
}

uint32_t
VpuUnit::loadUint32(const std::vector<uint8_t> &bytes, size_t offset) const
{
    panic_if(offset + sizeof(uint32_t) > bytes.size(),
             "%s: uint32 load out of range", name());
    uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

float
VpuUnit::loadFloat32(const std::vector<uint8_t> &bytes, size_t offset) const
{
    panic_if(offset + sizeof(float) > bytes.size(),
             "%s: float load out of range", name());
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void
VpuUnit::storeUint32(std::vector<uint8_t> &bytes, size_t offset,
                     uint32_t value) const
{
    panic_if(offset + sizeof(uint32_t) > bytes.size(),
             "%s: uint32 store out of range", name());
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void
VpuUnit::storeFloat32(std::vector<uint8_t> &bytes, size_t offset,
                      float value) const
{
    panic_if(offset + sizeof(float) > bytes.size(),
             "%s: float store out of range", name());
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

bool
VpuUnit::isLutOpcode(Opcode opcode) const
{
    return opcode == Opcode::VSqrt || opcode == Opcode::VExp ||
           opcode == Opcode::VSoftmax;
}

bool
VpuUnit::isLinearOpcode(Opcode opcode) const
{
    return !isLutOpcode(opcode) && opcode != Opcode::VLoad &&
           opcode != Opcode::VStore;
}

LutUnit::Operation
VpuUnit::lutOperation(Opcode opcode) const
{
    switch (opcode) {
      case Opcode::VSqrt:
        return LutUnit::Operation::Sqrt;
      case Opcode::VExp:
        return LutUnit::Operation::Exp;
      case Opcode::VSoftmax:
        return LutUnit::Operation::Softmax;
      default:
        panic("%s: opcode does not use LUT", name());
    }
}

VpuUnit::LocalBufferSlot &
VpuUnit::bufferSlot(const LocalAddr &addr)
{
    auto &buffers =
        addr.role == BufferRole::Input ? inputBuffers : outputBuffers;
    return buffers.at(addr.portId).at(addr.bufferIndex);
}

const VpuUnit::LocalBufferSlot &
VpuUnit::bufferSlot(const LocalAddr &addr) const
{
    const auto &buffers =
        addr.role == BufferRole::Input ? inputBuffers : outputBuffers;
    return buffers.at(addr.portId).at(addr.bufferIndex);
}

const std::vector<uint8_t> &
VpuUnit::sourceBytes(const VpuMacroState &state, size_t sourceIndex,
                     PortID logicalPort) const
{
    const Addr base =
        sourceIndex == 0 ? state.src0Addr :
        sourceIndex == 1 ? state.src1Addr : state.src2Addr;
    LocalAddr addr = decodeLocalAddr(
        base, BufferRole::Input, state.op.srcSpanBytes);
    addr.portId = logicalPort;
    const auto &slot = bufferSlot(addr);
    panic_if(!slot.valid || slot.bytes.size() < state.op.srcSpanBytes,
             "%s: missing input buffer data for port=%d buffer=%u",
             name(), logicalPort, addr.bufferIndex);
    return slot.bytes;
}

bool
VpuUnit::sourceReady(const VpuMacroState &state, size_t sourceIndex,
                     PortID logicalPort) const
{
    const Addr base =
        sourceIndex == 0 ? state.src0Addr :
        sourceIndex == 1 ? state.src1Addr : state.src2Addr;
    LocalAddr addr = decodeLocalAddr(
        base, BufferRole::Input, state.op.srcSpanBytes);
    addr.portId = logicalPort;
    const auto &slot = bufferSlot(addr);
    return slot.valid && slot.bytes.size() >= state.op.srcSpanBytes;
}

bool
VpuUnit::computeInputsReady(const VpuMacroState &state) const
{
    switch (state.op.opcode) {
      case Opcode::Exec:
        for (const PortID port : state.readPorts) {
            if (!sourceReady(state, 0, port)) {
                return false;
            }
        }
        for (const PortID port : state.writePorts) {
            if (!sourceReady(state, 0, port)) {
                return false;
            }
        }
        return true;
      case Opcode::VAdd:
      case Opcode::VSub:
      case Opcode::VMul:
      case Opcode::VDiv:
        for (size_t dstIndex = 0;
             dstIndex < state.writePorts.size(); ++dstIndex) {
            if (!sourceReady(state, 0, state.readPorts[dstIndex * 2]) ||
                !sourceReady(state, 1, state.readPorts[(dstIndex * 2) + 1])) {
                return false;
            }
        }
        return true;
      case Opcode::VFma:
        for (size_t dstIndex = 0;
             dstIndex < state.writePorts.size(); ++dstIndex) {
            if (!sourceReady(state, 0, state.readPorts[dstIndex * 3]) ||
                !sourceReady(state, 1, state.readPorts[(dstIndex * 3) + 1]) ||
                !sourceReady(state, 2, state.readPorts[(dstIndex * 3) + 2])) {
                return false;
            }
        }
        return true;
      case Opcode::VReduceSum:
      case Opcode::VReduceMax:
      case Opcode::VScale:
      case Opcode::VCvtI2F:
      case Opcode::VCvtF2I:
      case Opcode::VSqrt:
      case Opcode::VExp:
      case Opcode::VSoftmax:
        for (size_t dstIndex = 0;
             dstIndex < state.writePorts.size(); ++dstIndex) {
            if (!sourceReady(state, 0, state.readPorts[dstIndex])) {
                return false;
            }
        }
        return true;
      case Opcode::VLoad:
      case Opcode::VStore:
        return true;
    }
    return true;
}

bool
VpuUnit::storeSourceReady(const VpuMacroState &state) const
{
    const BufferRole srcRole = isOutputLocalAddr(state.src0Addr) ?
        BufferRole::Output : BufferRole::Input;
    const LocalAddr srcAddr = decodeLocalAddr(
        state.src0Addr, srcRole, state.op.dstSpanBytes);
    auto actualSrc = srcAddr;
    actualSrc.portId = state.writePorts.front();
    const auto &slot = bufferSlot(actualSrc);
    return slot.valid && slot.bytes.size() >= state.op.dstSpanBytes;
}

void
VpuUnit::writeResultBytes(const VpuMacroState &state, PortID dstPort,
                          const std::vector<uint8_t> &bytes) const
{
    LocalAddr addr = decodeLocalAddr(state.dstAddr, BufferRole::Output,
                                     state.op.dstSpanBytes);
    addr.portId = dstPort;
    auto &slot = const_cast<VpuUnit *>(this)->bufferSlot(addr);
    slot.bytes = bytes;
    slot.valid = true;
}

void
VpuUnit::appendStoreWhenReady(MacroCmdContext &macroCmd,
                              const VpuMacroState &state)
{
    if (!storeSourceReady(state)) {
        appendExecUop(macroCmd, 1);
        macroCmd.uopQueue.back().token =
            static_cast<uint64_t>(ExecToken::WaitStoreData);
        return;
    }

    const BufferRole srcRole = isOutputLocalAddr(state.src0Addr) ?
        BufferRole::Output : BufferRole::Input;
    const LocalAddr srcAddr = decodeLocalAddr(
        state.src0Addr, srcRole, state.op.dstSpanBytes);
    auto actualSrc = srcAddr;
    actualSrc.portId = state.writePorts.front();
    const auto &slot = bufferSlot(actualSrc);
    appendStoreUop(macroCmd, slotAddr(state.writePorts.front()),
                   state.op.dstSpanBytes,
                   std::vector<uint8_t>(slot.bytes.begin(),
                                        slot.bytes.begin() +
                                            state.op.dstSpanBytes));
}

void
VpuUnit::executeLegacy(VpuMacroState &state) const
{
    uint32_t signature = 0;
    for (const PortID port : state.readPorts) {
        const auto &bytes = sourceBytes(state, 0, port);
        signature += unpackWord(bytes);
    }

    for (const PortID port : state.writePorts) {
        const auto &currentBytes = sourceBytes(state, 0, port);
        const uint32_t current = unpackWord(currentBytes);
        const uint32_t value =
            current + signature + 0x10U + (static_cast<uint32_t>(port) + 1U);
        const auto outAddr =
            decodeLocalAddr(
                state.dstAddr, BufferRole::Output, sizeof(uint32_t));
        (void)outAddr;
        writeResultBytes(state, port, packWord(value));
    }
}

void
VpuUnit::executeBinary(const VpuMacroState &state) const
{
    for (size_t dstIndex = 0; dstIndex < state.writePorts.size(); ++dstIndex) {
        const PortID dstPort = state.writePorts[dstIndex];
        const PortID lhsPort = state.readPorts[dstIndex * 2];
        const PortID rhsPort = state.readPorts[(dstIndex * 2) + 1];
        const auto &lhsBytes = sourceBytes(state, 0, lhsPort);
        const auto &rhsBytes = sourceBytes(state, 1, rhsPort);
        std::vector<uint8_t> dstBytes(state.op.dstSpanBytes, 0);

        for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
            const size_t srcOffset = static_cast<size_t>(elem) *
                state.op.srcStrideBytes;
            const size_t dstOffset = static_cast<size_t>(elem) *
                state.op.dstStrideBytes;

            if (state.op.dataType == DataType::Int32) {
                const uint32_t lhs = loadUint32(lhsBytes, srcOffset);
                const uint32_t rhs = loadUint32(rhsBytes, srcOffset);
                uint32_t value = 0;
                switch (state.op.opcode) {
                  case Opcode::VAdd:
                    value = lhs + rhs;
                    break;
                  case Opcode::VSub:
                    value = lhs - rhs;
                    break;
                  case Opcode::VMul:
                    value = lhs * rhs;
                    break;
                  case Opcode::VDiv: {
                    const int32_t lhsSigned = static_cast<int32_t>(lhs);
                    const int32_t rhsSigned = static_cast<int32_t>(rhs);
                    int32_t quotient = 0;
                    if (rhsSigned != 0) {
                        quotient = lhsSigned / rhsSigned;
                    }
                    value = static_cast<uint32_t>(quotient);
                    break;
                  }
                  default:
                    panic("%s: unexpected integer binary opcode", name());
                }
                storeUint32(dstBytes, dstOffset, value);
            } else {
                const float lhs = loadFloat32(lhsBytes, srcOffset);
                const float rhs = loadFloat32(rhsBytes, srcOffset);
                float value = 0.0f;
                switch (state.op.opcode) {
                  case Opcode::VAdd:
                    value = lhs + rhs;
                    break;
                  case Opcode::VSub:
                    value = lhs - rhs;
                    break;
                  case Opcode::VMul:
                    value = lhs * rhs;
                    break;
                  case Opcode::VDiv:
                    value = lhs / rhs;
                    break;
                  default:
                    panic("%s: unexpected float binary opcode", name());
                }
                storeFloat32(dstBytes, dstOffset, value);
            }
        }

        writeResultBytes(state, dstPort, dstBytes);
    }
}

void
VpuUnit::executeUnary(const VpuMacroState &state) const
{
    for (size_t dstIndex = 0; dstIndex < state.writePorts.size(); ++dstIndex) {
        const PortID dstPort = state.writePorts[dstIndex];
        const PortID srcPort = state.readPorts[dstIndex];
        const auto &srcBytes = sourceBytes(state, 0, srcPort);
        std::vector<uint8_t> dstBytes(state.op.dstSpanBytes, 0);

        if (state.op.opcode == Opcode::VSoftmax) {
            std::vector<float> expValues(state.op.elemCount, 0.0f);
            float maxValue = -std::numeric_limits<float>::infinity();
            float sum = 0.0f;
            for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
                const size_t srcOffset = static_cast<size_t>(elem) *
                    state.op.srcStrideBytes;
                maxValue = std::max(maxValue,
                                    loadFloat32(srcBytes, srcOffset));
            }
            for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
                const size_t srcOffset = static_cast<size_t>(elem) *
                    state.op.srcStrideBytes;
                expValues[elem] = lut->evaluateExp(
                    loadFloat32(srcBytes, srcOffset) - maxValue);
                sum += expValues[elem];
            }
            for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
                const size_t dstOffset = static_cast<size_t>(elem) *
                    state.op.dstStrideBytes;
                storeFloat32(dstBytes, dstOffset, expValues[elem] / sum);
            }
            writeResultBytes(state, dstPort, dstBytes);
            continue;
        }

        for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
            const size_t srcOffset = static_cast<size_t>(elem) *
                state.op.srcStrideBytes;
            const size_t dstOffset = static_cast<size_t>(elem) *
                state.op.dstStrideBytes;

            switch (state.op.opcode) {
              case Opcode::VScale:
                if (state.op.dataType == DataType::Int32) {
                    const int32_t value = static_cast<int32_t>(
                        loadUint32(srcBytes, srcOffset));
                    const int32_t scalar =
                        static_cast<int32_t>(state.op.scalarBits);
                    storeUint32(dstBytes, dstOffset,
                                static_cast<uint32_t>(value * scalar));
                } else {
                    float scalar = 0.0f;
                    const float value = loadFloat32(srcBytes, srcOffset);
                    std::memcpy(&scalar, &state.op.scalarBits, sizeof(scalar));
                    storeFloat32(dstBytes, dstOffset, value * scalar);
                }
                break;
              case Opcode::VCvtI2F:
                storeFloat32(dstBytes, dstOffset, static_cast<float>(
                    static_cast<int32_t>(loadUint32(srcBytes, srcOffset))));
                break;
              case Opcode::VCvtF2I:
                storeUint32(dstBytes, dstOffset, static_cast<uint32_t>(
                    static_cast<int32_t>(std::trunc(
                        loadFloat32(srcBytes, srcOffset)))));
                break;
              case Opcode::VSqrt:
                storeFloat32(
                    dstBytes, dstOffset,
                    lut->evaluateSqrt(loadFloat32(srcBytes, srcOffset)));
                break;
              case Opcode::VExp:
                storeFloat32(
                    dstBytes, dstOffset,
                    lut->evaluateExp(loadFloat32(srcBytes, srcOffset)));
                break;
              default:
                panic("%s: unexpected unary opcode", name());
            }
        }

        writeResultBytes(state, dstPort, dstBytes);
    }
}

void
VpuUnit::executeFma(const VpuMacroState &state) const
{
    for (size_t dstIndex = 0; dstIndex < state.writePorts.size(); ++dstIndex) {
        const PortID dstPort = state.writePorts[dstIndex];
        const PortID src0Port = state.readPorts[dstIndex * 3];
        const PortID src1Port = state.readPorts[(dstIndex * 3) + 1];
        const PortID src2Port = state.readPorts[(dstIndex * 3) + 2];
        const auto &src0Bytes = sourceBytes(state, 0, src0Port);
        const auto &src1Bytes = sourceBytes(state, 1, src1Port);
        const auto &src2Bytes = sourceBytes(state, 2, src2Port);
        std::vector<uint8_t> dstBytes(state.op.dstSpanBytes, 0);

        for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
            const size_t srcOffset = static_cast<size_t>(elem) *
                state.op.srcStrideBytes;
            const size_t dstOffset = static_cast<size_t>(elem) *
                state.op.dstStrideBytes;
            const float value = std::fma(loadFloat32(src0Bytes, srcOffset),
                                         loadFloat32(src1Bytes, srcOffset),
                                         loadFloat32(src2Bytes, srcOffset));
            storeFloat32(dstBytes, dstOffset, value);
        }

        writeResultBytes(state, dstPort, dstBytes);
    }
}

void
VpuUnit::executeReduce(const VpuMacroState &state) const
{
    for (size_t dstIndex = 0; dstIndex < state.writePorts.size(); ++dstIndex) {
        const PortID dstPort = state.writePorts[dstIndex];
        const PortID srcPort = state.readPorts[dstIndex];
        const auto &srcBytes = sourceBytes(state, 0, srcPort);
        std::vector<uint8_t> dstBytes(sizeof(uint32_t), 0);

        if (state.op.dataType == DataType::Int32) {
            int32_t accum = 0;
            int32_t currentMax = std::numeric_limits<int32_t>::min();
            for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
                const size_t srcOffset = static_cast<size_t>(elem) *
                    state.op.srcStrideBytes;
                const int32_t value = static_cast<int32_t>(
                    loadUint32(srcBytes, srcOffset));
                accum += value;
                currentMax = std::max(currentMax, value);
            }
            const int32_t result =
                state.op.opcode == Opcode::VReduceSum ? accum : currentMax;
            storeUint32(dstBytes, 0, static_cast<uint32_t>(result));
        } else {
            float accum = 0.0f;
            float currentMax = -std::numeric_limits<float>::infinity();
            for (uint32_t elem = 0; elem < state.op.elemCount; ++elem) {
                const size_t srcOffset = static_cast<size_t>(elem) *
                    state.op.srcStrideBytes;
                const float value = loadFloat32(srcBytes, srcOffset);
                accum += value;
                currentMax = std::max(currentMax, value);
            }
            const float result =
                state.op.opcode == Opcode::VReduceSum ? accum : currentMax;
            storeFloat32(dstBytes, 0, result);
        }

        writeResultBytes(state, dstPort, dstBytes);
    }
}

void
VpuUnit::executeLoadStoreBypass(const VpuMacroState &state) const
{
    (void)state;
}

void
VpuUnit::executeVectorOp(VpuMacroState &state) const
{
    switch (state.op.opcode) {
      case Opcode::Exec:
        executeLegacy(state);
        break;
      case Opcode::VAdd:
      case Opcode::VSub:
      case Opcode::VMul:
      case Opcode::VDiv:
        executeBinary(state);
        break;
      case Opcode::VScale:
      case Opcode::VCvtI2F:
      case Opcode::VCvtF2I:
      case Opcode::VSqrt:
      case Opcode::VExp:
      case Opcode::VSoftmax:
        executeUnary(state);
        break;
      case Opcode::VFma:
        executeFma(state);
        break;
      case Opcode::VReduceSum:
      case Opcode::VReduceMax:
        executeReduce(state);
        break;
      case Opcode::VLoad:
      case Opcode::VStore:
        executeLoadStoreBypass(state);
        break;
    }
}

SpecializedExecutionUnit::MacroCmdKind
VpuUnit::classifyMacroCmd(const std::vector<uint8_t> &cmd) const
{
    switch (decodeOpcode(parseCmdFields(extractCmdWord(cmd)).opCode)) {
      case Opcode::VLoad:
        return MacroCmdKind::Load;
      case Opcode::VStore:
        return MacroCmdKind::Store;
      default:
        return MacroCmdKind::Exec;
    }
}

uint32_t
VpuUnit::classifyIssueQueue(const std::vector<uint8_t> &cmd,
                            MacroCmdKind kind) const
{
    if (kind == MacroCmdKind::Exec) {
        return 0;
    }

    const uint32_t maskWord = kind == MacroCmdKind::Load ?
        cmdWord(cmd, ReadMaskWord) : cmdWord(cmd, WriteMaskWord);
    const auto ports = decodeMask(maskWord);
    panic_if(ports.size() != 1,
             "%s: mem-side VPU commands must target exactly one SPM port",
             name());
    return 1 + ports.front();
}

void
VpuUnit::onMacroCmdBegin(MacroCmdContext &macroCmd)
{
    VpuMacroState state;
    state.op = decodeVectorOp(macroCmd);
    validateCommand(macroCmd, state);
    macroStates.emplace(macroCmd.macroCmdId, std::move(state));
}

void
VpuUnit::buildUops(MacroCmdContext &macroCmd)
{
    auto it = macroStates.find(macroCmd.macroCmdId);
    panic_if(it == macroStates.end(), "%s: missing VPU macro state", name());
    auto &state = it->second;

    switch (state.op.opcode) {
      case Opcode::VLoad: {
        const PortID spmPort = state.readPorts.front();
        appendLoadUop(macroCmd, slotAddr(spmPort), state.op.srcSpanBytes);
        break;
      }
      case Opcode::VStore:
        appendStoreWhenReady(macroCmd, state);
        break;
      default: {
        if (!computeInputsReady(state)) {
            appendExecUop(macroCmd, 1);
            macroCmd.uopQueue.back().token =
                static_cast<uint64_t>(ExecToken::WaitInputs);
        } else {
            const Tick latency = computeExecLatency(state);
            appendExecUop(macroCmd, latency);
            macroCmd.uopQueue.back().token =
                static_cast<uint64_t>(ExecToken::RunCompute);
        }
        break;
      }
    }

    if (macroCmd.uopQueue.empty()) {
        markEpiloguePending(macroCmd);
    }
}

void
VpuUnit::onMemUopComplete(MacroCmdContext &macroCmd,
                          const MemTxnContext &txn, PacketPtr pkt)
{
    auto it = macroStates.find(macroCmd.macroCmdId);
    panic_if(it == macroStates.end(), "%s: missing VPU macro state", name());
    auto &state = it->second;

    if (state.op.opcode == Opcode::VLoad) {
        LocalAddr dstAddr = decodeLocalAddr(state.dstAddr, BufferRole::Input,
                                            state.op.srcSpanBytes);
        dstAddr.portId = txn.portId;
        auto &slot = bufferSlot(dstAddr);
        slot.bytes.assign(pkt->getConstPtr<uint8_t>(),
                          pkt->getConstPtr<uint8_t>() + pkt->getSize());
        slot.valid = true;
    }

    markEpiloguePending(macroCmd);
}

void
VpuUnit::onExecUopComplete(MacroCmdContext &macroCmd,
                           const MicroOpContext &uop)
{
    (void)uop;
    auto it = macroStates.find(macroCmd.macroCmdId);
    panic_if(it == macroStates.end(), "%s: missing VPU macro state", name());
    auto &state = it->second;

    const ExecToken token = static_cast<ExecToken>(uop.token);
    if (state.op.opcode == Opcode::VStore ||
        token == ExecToken::WaitStoreData) {
        appendStoreWhenReady(macroCmd, state);
        if (macroCmd.uopQueue.empty()) {
            markEpiloguePending(macroCmd);
        }
        return;
    }

    if (token == ExecToken::WaitInputs) {
        if (!computeInputsReady(state)) {
            appendExecUop(macroCmd, 1);
            macroCmd.uopQueue.back().token =
                static_cast<uint64_t>(ExecToken::WaitInputs);
            return;
        }
        const Tick latency = computeExecLatency(state);
        appendExecUop(macroCmd, latency);
        macroCmd.uopQueue.back().token =
            static_cast<uint64_t>(ExecToken::RunCompute);
        return;
    }

    executeVectorOp(state);
    state.completedExecUops++;
    markEpiloguePending(macroCmd);
}

void
VpuUnit::onMacroCmdEnd(MacroCmdContext &macroCmd)
{
    auto it = macroStates.find(macroCmd.macroCmdId);
    panic_if(it == macroStates.end(), "%s: missing VPU macro state", name());
    const Opcode opcode = it->second.op.opcode;
    if (isLutOpcode(opcode)) {
        lut->noteCompletion(lutOperation(opcode), curTick());
    } else if (isLinearOpcode(opcode)) {
        lastLinearCompletionTickValue = curTick();
    }
    macroStates.erase(it);
}

uint64_t
VpuUnit::lutRequestCount() const
{
    return lut->requestCount();
}

uint64_t
VpuUnit::lutCommandCount() const
{
    return lut->commandCount();
}

Tick
VpuUnit::lastLinearExecuteLatency() const
{
    return lastLinearExecuteLatencyValue;
}

Tick
VpuUnit::lastLutExecuteLatency() const
{
    return lut->lastExecuteLatency();
}

Tick
VpuUnit::lastSoftmaxExecuteLatency() const
{
    return lut->lastSoftmaxExecuteLatency();
}

Tick
VpuUnit::lastLinearCompletionTick() const
{
    return lastLinearCompletionTickValue;
}

Tick
VpuUnit::lastLutCompletionTick() const
{
    return lut->lastCompletionTick();
}

Tick
VpuUnit::lastSoftmaxCompletionTick() const
{
    return lut->lastSoftmaxCompletionTick();
}

} // namespace gem5
