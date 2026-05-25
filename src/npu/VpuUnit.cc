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

#include <softfloat.h>

#include "base/logging.hh"
#include "debug/VPU.hh"

namespace gem5
{

namespace
{

constexpr uint32_t LocalRegionTagMask = 0xFF000000U;
constexpr uint8_t VAbsOpcodeValue = 0x9U;

inline uint32_t
divCeil(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

} // anonymous namespace

VpuUnit::VpuUnit(const VpuUnitParams &params)
    : SpecializedExecutionUnit(params), deviceId(params.device_id),
      lut(params.lut), numMemPorts(params.num_mem_side_ports),
      numInputPorts(params.num_input_ports),
      numOutputPorts(params.num_output_ports),
      inputBufferCount(params.input_buffer_count),
      outputBufferCount(params.output_buffer_count),
      localInputBase(params.local_input_base),
      localOutputBase(params.local_output_base),
      localBufferStride(params.local_buffer_stride),
      dlenBytes(params.dlen_bytes),
      int8CyclesPerDlen(params.int8_cycles_per_dlen),
      int16CyclesPerDlen(params.int16_cycles_per_dlen),
      int32CyclesPerDlen(params.int32_cycles_per_dlen),
      float16CyclesPerDlen(params.float16_cycles_per_dlen),
      float32CyclesPerDlen(params.float32_cycles_per_dlen),
      inputBuffers(inputBufferCount),
      outputBuffers(outputBufferCount)
{
    panic_if(lut == nullptr, "%s: lut must not be null", name());
    panic_if(numMemPorts == 0, "%s: mem ports must be non-zero", name());
    panic_if(numInputPorts < 2 || numOutputPorts == 0,
             "%s: VPU v2 requires at least two read ports and one write port",
             name());
    panic_if(inputBufferCount < 2 || outputBufferCount == 0,
             "%s: VPU v2 requires at least two input buffers and one output "
             "buffer",
             name());
    panic_if(localBufferStride == 0 || dlenBytes == 0,
             "%s: local buffer stride and dlen must be non-zero", name());
    panic_if((localBufferStride % dlenBytes) != 0,
             "%s: local buffer stride must contain an integer number of dlen "
             "chunks",
             name());
    panic_if(lut->dlenBytesValue() != dlenBytes,
             "%s: lut dlen_bytes (%u) must match VPU dlen_bytes (%u)",
             name(), lut->dlenBytesValue(), dlenBytes);
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
      case VAbsOpcodeValue:
        return static_cast<Opcode>(VAbsOpcodeValue);
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
      default:
        fatal("%s: unsupported VPU v2 opcode=%u", name(), opCode);
    }
}

VpuUnit::DataType
VpuUnit::decodeDataType(uint32_t raw) const
{
    switch (raw & 0xFU) {
      case static_cast<uint32_t>(DataType::Int8):
        return DataType::Int8;
      case static_cast<uint32_t>(DataType::Int16):
        return DataType::Int16;
      case static_cast<uint32_t>(DataType::Int32):
        return DataType::Int32;
      case static_cast<uint32_t>(DataType::UInt8):
        return DataType::UInt8;
      case static_cast<uint32_t>(DataType::UInt16):
        return DataType::UInt16;
      case static_cast<uint32_t>(DataType::UInt32):
        return DataType::UInt32;
      case static_cast<uint32_t>(DataType::Float16):
        return DataType::Float16;
      case static_cast<uint32_t>(DataType::Float32):
        return DataType::Float32;
      default:
        fatal("%s: unsupported VPU v2 data type %#x", name(), raw & 0xFU);
    }
}

VpuUnit::AxisPair
VpuUnit::decodeAxisField(uint32_t raw) const
{
    const uint32_t mode = raw >> 28U;
    panic_if(mode > 7U, "%s: unsupported axis packing mode=%u", name(), mode);

    const uint32_t wChunks = mode;
    const uint32_t cChunks = 7U - wChunks;
    const uint32_t payload = raw & 0x0FFFFFFFU;

    uint32_t wValue = 0U;
    uint32_t cValue = 0U;
    if (wChunks != 0U) {
        wValue = (payload >> (cChunks * 4U)) & ((1U << (wChunks * 4U)) - 1U);
    }
    if (cChunks != 0U) {
        cValue = payload & ((1U << (cChunks * 4U)) - 1U);
    }

    return {wValue + 1U, cValue + 1U};
}

VpuUnit::TensorDesc
VpuUnit::decodeTensor(const std::vector<uint8_t> &cmd, size_t addrWord) const
{
    return {
        cmdWord(cmd, addrWord),
        decodeAxisField(cmdWord(cmd, addrWord + 1U)),
        decodeAxisField(cmdWord(cmd, addrWord + 2U)),
    };
}

size_t
VpuUnit::elemSizeBytes(DataType dataType) const
{
    switch (dataType) {
      case DataType::Int8:
      case DataType::UInt8:
        return 1;
      case DataType::Int16:
      case DataType::UInt16:
      case DataType::Float16:
        return 2;
      case DataType::Int32:
      case DataType::UInt32:
      case DataType::Float32:
        return 4;
    }

    panic("%s: unreachable elemSizeBytes", name());
}

bool
VpuUnit::isFloatType(DataType dataType) const
{
    return dataType == DataType::Float16 || dataType == DataType::Float32;
}

bool
VpuUnit::isSignedType(DataType dataType) const
{
    return dataType == DataType::Int8 || dataType == DataType::Int16 ||
           dataType == DataType::Int32;
}

const char *
VpuUnit::opcodeName(Opcode opcode) const
{
    if (opcode == static_cast<Opcode>(VAbsOpcodeValue)) {
        return "VAbs";
    }

    switch (opcode) {
      case Opcode::VAdd:
        return "VAdd";
      case Opcode::VSub:
        return "VSub";
      case Opcode::VMul:
        return "VMul";
      case Opcode::VDiv:
        return "VDiv";
      case Opcode::VScale:
        return "VScale";
      case Opcode::VCvtI2F:
        return "VCvtI2F";
      case Opcode::VCvtF2I:
        return "VCvtF2I";
      case Opcode::VSqrt:
        return "VSqrt";
      case Opcode::VReduceSum:
        return "VReduceSum";
      case Opcode::VReduceMax:
        return "VReduceMax";
      case Opcode::VLoad:
        return "VLoad";
      case Opcode::VStore:
        return "VStore";
      case Opcode::VExp:
        return "VExp";
    }

    panic("%s: unreachable opcodeName", name());
}

const char *
VpuUnit::dataTypeName(DataType dataType) const
{
    switch (dataType) {
      case DataType::Int8:
        return "Int8";
      case DataType::Int16:
        return "Int16";
      case DataType::Int32:
        return "Int32";
      case DataType::UInt8:
        return "UInt8";
      case DataType::UInt16:
        return "UInt16";
      case DataType::UInt32:
        return "UInt32";
      case DataType::Float16:
        return "Float16";
      case DataType::Float32:
        return "Float32";
    }

    panic("%s: unreachable dataTypeName", name());
}

VpuUnit::DecodedVectorOp
VpuUnit::decodeVectorOp(const MacroCmdContext &macroCmd) const
{
    DecodedVectorOp op;
    op.opcode = decodeOpcode(macroCmd.fields.opCode);

    const uint32_t format = cmdWord(macroCmd.cmd, FormatWord);
    op.dstType = decodeDataType(format);
    op.src0Type = decodeDataType(format >> 4U);
    op.src1Type = decodeDataType(format >> 8U);

    switch ((format >> 20U) & 0xFU) {
      case 0U:
        op.layoutOrder = LayoutOrder::WC;
        break;
      case 1U:
        op.layoutOrder = LayoutOrder::CW;
        break;
      default:
        fatal("%s: unsupported layout order=%u", name(),
              (format >> 20U) & 0xFU);
    }

    op.wLayoutElems = 1U << ((format >> 12U) & 0xFU);
    op.cLayoutElems = 1U << ((format >> 16U) & 0xFU);
    op.scalarBits = cmdWord(macroCmd.cmd, Extra0Word);
    op.dstElemSize = elemSizeBytes(op.dstType);
    op.src0ElemSize = elemSizeBytes(op.src0Type);
    op.src1ElemSize = elemSizeBytes(op.src1Type);
    op.isReduce = op.opcode == Opcode::VReduceSum ||
        op.opcode == Opcode::VReduceMax;
    op.hasSrc1 = op.opcode == Opcode::VAdd || op.opcode == Opcode::VSub ||
        op.opcode == Opcode::VMul || op.opcode == Opcode::VDiv;
    return op;
}

bool
VpuUnit::isComputeOpcode(Opcode opcode) const
{
    return opcode != Opcode::VLoad && opcode != Opcode::VStore;
}

bool
VpuUnit::isLutOpcode(Opcode opcode) const
{
    return opcode == Opcode::VSqrt || opcode == Opcode::VExp;
}

bool
VpuUnit::isLinearOpcode(Opcode opcode) const
{
    return !isLutOpcode(opcode) && opcode != Opcode::VLoad &&
           opcode != Opcode::VStore;
}

bool
VpuUnit::isSpmAddr(Addr addr) const
{
    return addr >= SpmBase && addr < localInputBase;
}

bool
VpuUnit::isInputLocalAddr(Addr addr) const
{
    return (addr & LocalRegionTagMask) == (localInputBase & LocalRegionTagMask);
}

bool
VpuUnit::isOutputLocalAddr(Addr addr) const
{
    return (addr & LocalRegionTagMask) == (localOutputBase & LocalRegionTagMask);
}

VpuUnit::LocalAddr
VpuUnit::decodeLocalAddr(Addr addr, BufferRole expected, size_t accessSize) const
{
    const Addr base = expected == BufferRole::Input ? localInputBase :
        localOutputBase;
    const uint32_t bufferCount = expected == BufferRole::Input ?
        inputBufferCount : outputBufferCount;

    panic_if((addr & LocalRegionTagMask) != (base & LocalRegionTagMask),
             "%s: address %#llx is not in expected local region", name(),
             static_cast<unsigned long long>(addr));

    const Addr offset = addr - base;
    const uint32_t bufferIndex = offset / localBufferStride;
    const size_t intraOffset = offset % localBufferStride;
    panic_if(bufferIndex >= bufferCount,
             "%s: local buffer index %u out of range", name(), bufferIndex);
    panic_if(intraOffset + accessSize > localBufferStride,
             "%s: local-buffer access exceeds slot stride", name());
    return {expected, bufferIndex, intraOffset};
}

PortID
VpuUnit::decodeSpmPort(Addr addr, size_t accessSize) const
{
    panic_if(!isSpmAddr(addr), "%s: address %#llx is not in SPM", name(),
             static_cast<unsigned long long>(addr));
    (void)accessSize;
    const Addr offset = addr - SpmBase;
    const PortID port = (offset / SpmSlotStride) % numMemPorts;
    return port;
}

uint32_t
VpuUnit::layoutLowestDim(const DecodedVectorOp &op) const
{
    return op.layoutOrder == LayoutOrder::WC ? op.cLayoutElems : op.wLayoutElems;
}

size_t
VpuUnit::tileElems(const DecodedVectorOp &op) const
{
    return static_cast<size_t>(op.wLayoutElems) * op.cLayoutElems;
}

Cycles
VpuUnit::dtypeCyclesPerDlen(DataType dataType) const
{
    switch (dataType) {
      case DataType::Int8:
      case DataType::UInt8:
        return int8CyclesPerDlen;
      case DataType::Int16:
      case DataType::UInt16:
        return int16CyclesPerDlen;
      case DataType::Int32:
      case DataType::UInt32:
        return int32CyclesPerDlen;
      case DataType::Float16:
        return float16CyclesPerDlen;
      case DataType::Float32:
        return float32CyclesPerDlen;
    }

    panic("%s: unreachable dtypeCyclesPerDlen", name());
}

uint32_t
VpuUnit::workDlenChunks(const VpuMacroState &state) const
{
    size_t workBytes = static_cast<size_t>(state.dst.shape.w) *
        state.dst.shape.c * state.op.dstElemSize;

    if (state.op.isReduce) {
        workBytes = static_cast<size_t>(state.src0.shape.w) *
            state.src0.shape.c * state.op.src0ElemSize;
    }

    return std::max<uint32_t>(
        1U, static_cast<uint32_t>((workBytes + dlenBytes - 1U) / dlenBytes));
}

size_t
VpuUnit::tensorSpanBytes(const TensorDesc &tensor, size_t elemSize,
                         const DecodedVectorOp &op) const
{
    const uint32_t wTiles = divCeil(tensor.shape.w, op.wLayoutElems);
    const uint32_t cTiles = divCeil(tensor.shape.c, op.cLayoutElems);
    const size_t lastTile =
        (static_cast<size_t>(wTiles - 1U) * tensor.stride.w) +
        (static_cast<size_t>(cTiles - 1U) * tensor.stride.c);
    return (lastTile + 1U) * tileElems(op) * elemSize;
}

size_t
VpuUnit::tensorElemOffset(const TensorDesc &tensor, const DecodedVectorOp &op,
                          uint32_t w, uint32_t c, size_t elemSize) const
{
    const uint32_t wOuter = w / op.wLayoutElems;
    const uint32_t cOuter = c / op.cLayoutElems;
    const uint32_t wInner = w % op.wLayoutElems;
    const uint32_t cInner = c % op.cLayoutElems;
    const size_t tileBase =
        (static_cast<size_t>(wOuter) * tensor.stride.w) +
        (static_cast<size_t>(cOuter) * tensor.stride.c);
    const size_t innerIndex = op.layoutOrder == LayoutOrder::WC ?
        (static_cast<size_t>(wInner) * op.cLayoutElems) + cInner :
        (static_cast<size_t>(cInner) * op.wLayoutElems) + wInner;
    return (tileBase * tileElems(op) + innerIndex) * elemSize;
}

VpuUnit::AxisPair
VpuUnit::broadcastShape(const TensorDesc &src0, const TensorDesc &src1) const
{
    auto broadcastAxis = [this](uint32_t lhs, uint32_t rhs, const char *axis) {
        panic_if(lhs != rhs && lhs != 1U && rhs != 1U,
                 "%s: incompatible broadcast on %s axis (%u vs %u)", name(),
                 axis, lhs, rhs);
        return std::max(lhs, rhs);
    };

    return {
        broadcastAxis(src0.shape.w, src1.shape.w, "W"),
        broadcastAxis(src0.shape.c, src1.shape.c, "C"),
    };
}

void
VpuUnit::validateTensor(const TensorDesc &tensor, size_t elemSize,
                        const DecodedVectorOp &op, const char *label) const
{
    panic_if(tensor.shape.w == 0U || tensor.shape.c == 0U ||
                 tensor.stride.w == 0U || tensor.stride.c == 0U,
             "%s: %s shape/stride must be non-zero", name(), label);
    panic_if((layoutLowestDim(op) * elemSize) != dlenBytes,
             "%s: %s violates dlen rule (lowest layout dim %u * elem bytes %zu "
             "!= %u)",
             name(), label, layoutLowestDim(op), elemSize, dlenBytes);
    panic_if(tensor.shape.w > 1U && (tensor.shape.w % op.wLayoutElems) != 0U,
             "%s: %s W extent must align with layout W", name(), label);
    panic_if(tensor.shape.c > 1U && (tensor.shape.c % op.cLayoutElems) != 0U,
             "%s: %s C extent must align with layout C", name(), label);

    const size_t spanBytes = tensorSpanBytes(tensor, elemSize, op);
    const size_t tileBytes = tileElems(op) * elemSize;
    if (isInputLocalAddr(tensor.addr) || isOutputLocalAddr(tensor.addr)) {
        const Addr base = isInputLocalAddr(tensor.addr) ? localInputBase :
            localOutputBase;
        panic_if(((tensor.addr - base) % tileBytes) != 0U,
                 "%s: %s address must be layout aligned", name(), label);
        panic_if(((tensor.addr - base) % localBufferStride) + spanBytes >
                     localBufferStride,
                 "%s: %s span exceeds local buffer slot", name(), label);
    } else if (isSpmAddr(tensor.addr)) {
        panic_if(((tensor.addr - SpmBase) % tileBytes) != 0U,
                 "%s: %s SPM address must be layout aligned", name(), label);
        decodeSpmPort(tensor.addr, spanBytes);
    } else {
        panic("%s: %s address %#llx is neither local nor SPM", name(), label,
              static_cast<unsigned long long>(tensor.addr));
    }
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

    state.dst = decodeTensor(macroCmd.cmd, DstAddrWord);
    state.src0 = decodeTensor(macroCmd.cmd, Src0AddrWord);
    state.src1 = decodeTensor(macroCmd.cmd, Src1AddrWord);

    validateTensor(state.dst, state.op.dstElemSize, state.op, "dst");
    validateTensor(state.src0, state.op.src0ElemSize, state.op, "src0");
    if (state.op.hasSrc1) {
        validateTensor(state.src1, state.op.src1ElemSize, state.op, "src1");
    }

    state.dstSpanBytes = tensorSpanBytes(state.dst, state.op.dstElemSize,
                                         state.op);
    state.src0SpanBytes = tensorSpanBytes(state.src0, state.op.src0ElemSize,
                                          state.op);
    state.src1SpanBytes = state.op.hasSrc1 ?
        tensorSpanBytes(state.src1, state.op.src1ElemSize, state.op) : 0U;

    if (state.op.opcode == Opcode::VLoad) {
        panic_if(!isSpmAddr(state.src0.addr),
                 "%s: VLOAD source must be in SPM", name());
        panic_if(!isInputLocalAddr(state.dst.addr),
                 "%s: VLOAD destination must be input local buffer", name());
        panic_if(state.dst.shape.w != state.src0.shape.w ||
                     state.dst.shape.c != state.src0.shape.c,
                 "%s: VLOAD source/destination shapes must match", name());
        return;
    }

    if (state.op.opcode == Opcode::VStore) {
        panic_if(!isSpmAddr(state.dst.addr),
                 "%s: VSTORE destination must be in SPM", name());
        panic_if(!isInputLocalAddr(state.src0.addr) &&
                     !isOutputLocalAddr(state.src0.addr),
                 "%s: VSTORE source must be local", name());
        panic_if(state.dst.shape.w != state.src0.shape.w ||
                     state.dst.shape.c != state.src0.shape.c,
                 "%s: VSTORE source/destination shapes must match", name());
        return;
    }

    panic_if(!isOutputLocalAddr(state.dst.addr) && !isSpmAddr(state.dst.addr),
             "%s: compute destination must be output local buffer or SPM", name());
    panic_if(!isInputLocalAddr(state.src0.addr) && !isSpmAddr(state.src0.addr),
             "%s: compute src0 must be input local buffer or SPM", name());
    if (state.op.hasSrc1) {
        panic_if(!isInputLocalAddr(state.src1.addr) && !isSpmAddr(state.src1.addr),
                 "%s: compute src1 must be input local buffer or SPM", name());
    }

    state.src0InSpm = isSpmAddr(state.src0.addr);
    state.src1InSpm = state.op.hasSrc1 && isSpmAddr(state.src1.addr);
    state.dstInSpm = isSpmAddr(state.dst.addr);
    state.src0Loaded = !state.src0InSpm;
    state.src1Loaded = !state.op.hasSrc1 || !state.src1InSpm;

    if (state.op.opcode == Opcode::VCvtI2F) {
        panic_if(!isFloatType(state.op.dstType) || isFloatType(state.op.src0Type),
                 "%s: VCvtI2F requires integer src0 and floating dst", name());
    } else if (state.op.opcode == Opcode::VCvtF2I) {
        panic_if(isFloatType(state.op.dstType) || !isFloatType(state.op.src0Type),
                 "%s: VCvtF2I requires floating src0 and integer dst", name());
    } else if (state.op.opcode == Opcode::VSqrt ||
               state.op.opcode == Opcode::VExp) {
        panic_if(!isFloatType(state.op.dstType) ||
                     !isFloatType(state.op.src0Type),
                 "%s: VSQRT/VEXP require floating operands", name());
    } else if (state.op.opcode == static_cast<Opcode>(VAbsOpcodeValue)) {
        panic_if(state.op.dstType != state.op.src0Type,
                 "%s: VABS requires dst/src0 dtype match",
                 name());
    } else {
        panic_if(state.op.dstType != state.op.src0Type,
                 "%s: dst and src0 dtypes must match", name());
        if (state.op.hasSrc1) {
            panic_if(state.op.src1Type != state.op.src0Type,
                     "%s: binary sources must use the same dtype", name());
        }
    }

    if (state.op.isReduce) {
        const bool hasWLayout = state.op.wLayoutElems > 1U;
        const bool hasCLayout = state.op.cLayoutElems > 1U;
        panic_if(hasWLayout && hasCLayout,
                 "%s: reduce ops only allow one layout axis", name());

        AxisPair expected{1U, 1U};
        if (hasWLayout) {
            expected = {state.src0.shape.w, 1U};
        } else if (hasCLayout) {
            expected = {1U, state.src0.shape.c};
        }

        panic_if(state.dst.shape.w != expected.w || state.dst.shape.c != expected.c,
                 "%s: reduce destination shape must match reduction result", name());
    } else if (state.op.hasSrc1) {
        const AxisPair expected = broadcastShape(state.src0, state.src1);
        panic_if(state.dst.shape.w != expected.w || state.dst.shape.c != expected.c,
                 "%s: binary destination shape must match broadcast result", name());
    } else {
        panic_if(state.dst.shape.w != state.src0.shape.w ||
                     state.dst.shape.c != state.src0.shape.c,
                 "%s: unary destination shape must match src0", name());
    }
}

Tick
VpuUnit::computeExecLatency(const VpuMacroState &state)
{
    const uint32_t dlenChunks = workDlenChunks(state);
    Cycles perDlenCycles = dtypeCyclesPerDlen(state.op.dstType);
    perDlenCycles = std::max(perDlenCycles, dtypeCyclesPerDlen(state.op.src0Type));
    if (state.op.hasSrc1) {
        perDlenCycles = std::max(perDlenCycles,
                                 dtypeCyclesPerDlen(state.op.src1Type));
    }

    Tick extraLatency = 0;
    if (isLutOpcode(state.op.opcode)) {
        const size_t workBytes = static_cast<size_t>(state.src0.shape.w) *
            state.src0.shape.c * state.op.src0ElemSize;
        extraLatency = lut->reserve(lutOperation(state.op.opcode), workBytes,
                                    curTick());
    }

    const Tick totalLatency = debugProcessLatency +
        (clockPeriod() * (perDlenCycles * dlenChunks)) + extraLatency;
    if (isLinearOpcode(state.op.opcode)) {
        lastLinearExecuteLatencyValue = totalLatency;
    }
    return totalLatency;
}

LutUnit::Operation
VpuUnit::lutOperation(Opcode opcode) const
{
    switch (opcode) {
      case Opcode::VSqrt:
        return LutUnit::Operation::Sqrt;
      case Opcode::VExp:
        return LutUnit::Operation::Exp;
      default:
        panic("%s: opcode does not use LUT", name());
    }
}

VpuUnit::LocalBufferSlot &
VpuUnit::bufferSlot(const LocalAddr &addr)
{
    auto &buffers = addr.role == BufferRole::Input ? inputBuffers : outputBuffers;
    return buffers.at(addr.bufferIndex);
}

const VpuUnit::LocalBufferSlot &
VpuUnit::bufferSlot(const LocalAddr &addr) const
{
    const auto &buffers = addr.role == BufferRole::Input ?
        inputBuffers : outputBuffers;
    return buffers.at(addr.bufferIndex);
}

const std::vector<uint8_t> &
VpuUnit::sourceBytes(const TensorDesc &tensor, const DecodedVectorOp &op,
                     size_t spanBytes) const
{
    const BufferRole role = isInputLocalAddr(tensor.addr) ?
        BufferRole::Input : BufferRole::Output;
    const LocalAddr addr = decodeLocalAddr(tensor.addr, role, spanBytes);
    const auto &slot = bufferSlot(addr);
    panic_if(!slot.valid || slot.bytes.size() < addr.offset + spanBytes,
             "%s: missing local buffer data", name());
    return slot.bytes;
}

const std::vector<uint8_t> &
VpuUnit::macroSourceBytes(const VpuMacroState &state, bool secondSource) const
{
    if (secondSource) {
        return state.src1InSpm ? state.src1Bytes :
            sourceBytes(state.src1, state.op, state.src1SpanBytes);
    }
    return state.src0InSpm ? state.src0Bytes :
        sourceBytes(state.src0, state.op, state.src0SpanBytes);
}

bool
VpuUnit::sourceReady(const TensorDesc &tensor, BufferRole role,
                     size_t spanBytes) const
{
    const LocalAddr addr = decodeLocalAddr(tensor.addr, role, spanBytes);
    const auto &slot = bufferSlot(addr);
    return slot.valid && slot.bytes.size() >= addr.offset + spanBytes;
}

bool
VpuUnit::storeSourceReady(const VpuMacroState &state) const
{
    const BufferRole role = isOutputLocalAddr(state.src0.addr) ?
        BufferRole::Output : BufferRole::Input;
    return sourceReady(state.src0, role, state.src0SpanBytes);
}

void
VpuUnit::writeResultBytes(const TensorDesc &dst, const std::vector<uint8_t> &bytes,
                          size_t spanBytes) const
{
    const LocalAddr addr = decodeLocalAddr(dst.addr, BufferRole::Output, spanBytes);
    auto &slot = const_cast<VpuUnit *>(this)->bufferSlot(addr);
    const size_t required = addr.offset + spanBytes;
    if (slot.bytes.size() < required) {
        slot.bytes.resize(required, 0);
    }
    std::copy(bytes.begin(), bytes.begin() + spanBytes,
              slot.bytes.begin() + addr.offset);
    slot.valid = true;
}

uint64_t
VpuUnit::loadUnsignedValue(const std::vector<uint8_t> &bytes, size_t offset,
                           size_t elemSize) const
{
    panic_if(offset + elemSize > bytes.size(), "%s: unsigned load out of range",
             name());
    uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, elemSize);
    return value;
}

int64_t
VpuUnit::loadSignedValue(const std::vector<uint8_t> &bytes, size_t offset,
                         size_t elemSize) const
{
    const uint64_t raw = loadUnsignedValue(bytes, offset, elemSize);
    switch (elemSize) {
      case 1:
        return static_cast<int8_t>(raw);
      case 2:
        return static_cast<int16_t>(raw);
      case 4:
        return static_cast<int32_t>(raw);
      default:
        panic("%s: unsupported signed element size %zu", name(), elemSize);
    }
}

double
VpuUnit::loadFloatValue(const std::vector<uint8_t> &bytes, size_t offset,
                        DataType dataType) const
{
    if (dataType == DataType::Float32) {
        float value = 0.0f;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }

    uint16_t raw = 0;
    std::memcpy(&raw, bytes.data() + offset, sizeof(raw));
    const float32_t converted = f16_to_f32(float16_t{raw});
    float value = 0.0f;
    std::memcpy(&value, &converted.v, sizeof(value));
    return value;
}

void
VpuUnit::storeUnsignedValue(std::vector<uint8_t> &bytes, size_t offset,
                            uint64_t value, size_t elemSize) const
{
    panic_if(offset + elemSize > bytes.size(), "%s: unsigned store out of range",
             name());
    std::memcpy(bytes.data() + offset, &value, elemSize);
}

void
VpuUnit::storeSignedValue(std::vector<uint8_t> &bytes, size_t offset,
                          int64_t value, size_t elemSize) const
{
    switch (elemSize) {
      case 1: {
        const int8_t narrowed = value;
        std::memcpy(bytes.data() + offset, &narrowed, sizeof(narrowed));
        break;
      }
      case 2: {
        const int16_t narrowed = value;
        std::memcpy(bytes.data() + offset, &narrowed, sizeof(narrowed));
        break;
      }
      case 4: {
        const int32_t narrowed = value;
        std::memcpy(bytes.data() + offset, &narrowed, sizeof(narrowed));
        break;
      }
      default:
        panic("%s: unsupported signed element size %zu", name(), elemSize);
    }
}

void
VpuUnit::storeFloatValue(std::vector<uint8_t> &bytes, size_t offset,
                         double value, DataType dataType) const
{
    if (dataType == DataType::Float32) {
        const float narrowed = value;
        std::memcpy(bytes.data() + offset, &narrowed, sizeof(narrowed));
        return;
    }

    const float asFloat = value;
    uint32_t raw32 = 0;
    std::memcpy(&raw32, &asFloat, sizeof(raw32));
    const float16_t converted = f32_to_f16(float32_t{raw32});
    std::memcpy(bytes.data() + offset, &converted.v, sizeof(converted.v));
}

void
VpuUnit::executeBinary(VpuMacroState &state) const
{
    const auto &src0Bytes = macroSourceBytes(state, false);
    const auto &src1Bytes = macroSourceBytes(state, true);
    std::vector<uint8_t> dstBytes(state.dstSpanBytes, 0);

    for (uint32_t w = 0; w < state.dst.shape.w; ++w) {
        for (uint32_t c = 0; c < state.dst.shape.c; ++c) {
            const uint32_t src0W = state.src0.shape.w == 1U ? 0U : w;
            const uint32_t src0C = state.src0.shape.c == 1U ? 0U : c;
            const uint32_t src1W = state.src1.shape.w == 1U ? 0U : w;
            const uint32_t src1C = state.src1.shape.c == 1U ? 0U : c;
            const size_t src0Offset = tensorElemOffset(
                state.src0, state.op, src0W, src0C, state.op.src0ElemSize);
            const size_t src1Offset = tensorElemOffset(
                state.src1, state.op, src1W, src1C, state.op.src1ElemSize);
            const size_t dstOffset = tensorElemOffset(
                state.dst, state.op, w, c, state.op.dstElemSize);

            if (isFloatType(state.op.dstType)) {
                const double lhs = loadFloatValue(src0Bytes, src0Offset,
                                                  state.op.src0Type);
                const double rhs = loadFloatValue(src1Bytes, src1Offset,
                                                  state.op.src1Type);
                double value = 0.0;
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
                    panic("%s: unexpected floating binary opcode", name());
                }
                storeFloatValue(dstBytes, dstOffset, value, state.op.dstType);
            } else if (isSignedType(state.op.dstType)) {
                const int64_t lhs = loadSignedValue(src0Bytes, src0Offset,
                                                    state.op.src0ElemSize);
                const int64_t rhs = loadSignedValue(src1Bytes, src1Offset,
                                                    state.op.src1ElemSize);
                int64_t value = 0;
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
                    value = rhs == 0 ? 0 : lhs / rhs;
                    break;
                  default:
                    panic("%s: unexpected signed binary opcode", name());
                }
                storeSignedValue(dstBytes, dstOffset, value, state.op.dstElemSize);
            } else {
                const uint64_t lhs = loadUnsignedValue(src0Bytes, src0Offset,
                                                       state.op.src0ElemSize);
                const uint64_t rhs = loadUnsignedValue(src1Bytes, src1Offset,
                                                       state.op.src1ElemSize);
                uint64_t value = 0;
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
                    value = rhs == 0 ? 0 : lhs / rhs;
                    break;
                  default:
                    panic("%s: unexpected unsigned binary opcode", name());
                }
                storeUnsignedValue(dstBytes, dstOffset, value, state.op.dstElemSize);
            }
        }
    }

    if (state.dstInSpm) {
        state.resultBytes = std::move(dstBytes);
        state.resultReady = true;
    } else {
        writeResultBytes(state.dst, dstBytes, state.dstSpanBytes);
        state.resultReady = true;
    }
}

void
VpuUnit::executeUnary(VpuMacroState &state) const
{
    const auto &src0Bytes = macroSourceBytes(state, false);
    std::vector<uint8_t> dstBytes(state.dstSpanBytes, 0);

    for (uint32_t w = 0; w < state.dst.shape.w; ++w) {
        for (uint32_t c = 0; c < state.dst.shape.c; ++c) {
            const size_t srcOffset = tensorElemOffset(
                state.src0, state.op, w, c, state.op.src0ElemSize);
            const size_t dstOffset = tensorElemOffset(
                state.dst, state.op, w, c, state.op.dstElemSize);

            if (state.op.opcode == static_cast<Opcode>(VAbsOpcodeValue)) {
                if (isFloatType(state.op.dstType)) {
                    storeFloatValue(dstBytes, dstOffset,
                                    std::fabs(loadFloatValue(
                                        src0Bytes, srcOffset,
                                        state.op.src0Type)),
                                    state.op.dstType);
                } else if (isSignedType(state.op.dstType)) {
                    const int64_t value = loadSignedValue(
                        src0Bytes, srcOffset, state.op.src0ElemSize);
                    storeSignedValue(dstBytes, dstOffset,
                                     value < 0 ? -value : value,
                                     state.op.dstElemSize);
                } else {
                    storeUnsignedValue(dstBytes, dstOffset,
                                       loadUnsignedValue(
                                           src0Bytes, srcOffset,
                                           state.op.src0ElemSize),
                                       state.op.dstElemSize);
                }
                continue;
            }

            switch (state.op.opcode) {
              case Opcode::VScale:
                if (isFloatType(state.op.dstType)) {
                    float scalar = 0.0f;
                    std::memcpy(&scalar, &state.op.scalarBits, sizeof(scalar));
                    storeFloatValue(dstBytes, dstOffset,
                                    loadFloatValue(src0Bytes, srcOffset,
                                                   state.op.src0Type) * scalar,
                                    state.op.dstType);
                } else if (isSignedType(state.op.dstType)) {
                    storeSignedValue(dstBytes, dstOffset,
                                     loadSignedValue(src0Bytes, srcOffset,
                                                     state.op.src0ElemSize) *
                                         static_cast<int32_t>(state.op.scalarBits),
                                     state.op.dstElemSize);
                } else {
                    storeUnsignedValue(dstBytes, dstOffset,
                                       loadUnsignedValue(src0Bytes, srcOffset,
                                                         state.op.src0ElemSize) *
                                           state.op.scalarBits,
                                       state.op.dstElemSize);
                }
                break;
              case Opcode::VCvtI2F:
                if (isSignedType(state.op.src0Type)) {
                    storeFloatValue(dstBytes, dstOffset,
                                    loadSignedValue(src0Bytes, srcOffset,
                                                    state.op.src0ElemSize),
                                    state.op.dstType);
                } else {
                    storeFloatValue(dstBytes, dstOffset,
                                    loadUnsignedValue(src0Bytes, srcOffset,
                                                      state.op.src0ElemSize),
                                    state.op.dstType);
                }
                break;
              case Opcode::VCvtF2I: {
                const double value = std::trunc(
                    loadFloatValue(src0Bytes, srcOffset, state.op.src0Type));
                if (isSignedType(state.op.dstType)) {
                    storeSignedValue(dstBytes, dstOffset,
                                     static_cast<int64_t>(value),
                                     state.op.dstElemSize);
                } else {
                    storeUnsignedValue(dstBytes, dstOffset,
                                       static_cast<uint64_t>(
                                           std::max(0.0, value)),
                                       state.op.dstElemSize);
                }
                break;
              }
              case Opcode::VSqrt:
                storeFloatValue(dstBytes, dstOffset,
                                lut->evaluateSqrt(loadFloatValue(
                                    src0Bytes, srcOffset, state.op.src0Type)),
                                state.op.dstType);
                break;
              case Opcode::VExp:
                storeFloatValue(dstBytes, dstOffset,
                                lut->evaluateExp(loadFloatValue(
                                    src0Bytes, srcOffset, state.op.src0Type)),
                                state.op.dstType);
                break;
              default:
                panic("%s: unexpected unary opcode", name());
            }
        }
    }

    if (state.dstInSpm) {
        state.resultBytes = std::move(dstBytes);
        state.resultReady = true;
    } else {
        writeResultBytes(state.dst, dstBytes, state.dstSpanBytes);
        state.resultReady = true;
    }
}

void
VpuUnit::executeReduce(VpuMacroState &state) const
{
    const auto &src0Bytes = macroSourceBytes(state, false);
    std::vector<uint8_t> dstBytes(state.dstSpanBytes, 0);

    const bool reduceAlongW = state.op.cLayoutElems > 1U;
    const bool reduceAlongC = state.op.wLayoutElems > 1U;

    const uint32_t outW = reduceAlongW ? 1U : state.src0.shape.w;
    const uint32_t outC = reduceAlongC ? 1U : state.src0.shape.c;

    for (uint32_t w = 0; w < outW; ++w) {
        for (uint32_t c = 0; c < outC; ++c) {
            const size_t dstOffset = tensorElemOffset(
                state.dst, state.op, w, c, state.op.dstElemSize);

            if (isFloatType(state.op.dstType)) {
                double accum = 0.0;
                double currentMax = -std::numeric_limits<double>::infinity();
                const uint32_t reduceLimit = reduceAlongW ? state.src0.shape.w :
                    (reduceAlongC ? state.src0.shape.c :
                                    state.src0.shape.w * state.src0.shape.c);
                for (uint32_t idx = 0; idx < reduceLimit; ++idx) {
                    const uint32_t srcW = reduceAlongW ? idx :
                        (reduceAlongC ? w : (idx / state.src0.shape.c));
                    const uint32_t srcC = reduceAlongW ? c :
                        (reduceAlongC ? idx : (idx % state.src0.shape.c));
                    const size_t srcOffset = tensorElemOffset(
                        state.src0, state.op, srcW, srcC, state.op.src0ElemSize);
                    const double value = loadFloatValue(
                        src0Bytes, srcOffset, state.op.src0Type);
                    accum += value;
                    currentMax = std::max(currentMax, value);
                }
                storeFloatValue(dstBytes, dstOffset,
                                state.op.opcode == Opcode::VReduceSum ?
                                accum : currentMax,
                                state.op.dstType);
            } else if (isSignedType(state.op.dstType)) {
                int64_t accum = 0;
                int64_t currentMax = std::numeric_limits<int64_t>::min();
                const uint32_t reduceLimit = reduceAlongW ? state.src0.shape.w :
                    (reduceAlongC ? state.src0.shape.c :
                                    state.src0.shape.w * state.src0.shape.c);
                for (uint32_t idx = 0; idx < reduceLimit; ++idx) {
                    const uint32_t srcW = reduceAlongW ? idx :
                        (reduceAlongC ? w : (idx / state.src0.shape.c));
                    const uint32_t srcC = reduceAlongW ? c :
                        (reduceAlongC ? idx : (idx % state.src0.shape.c));
                    const size_t srcOffset = tensorElemOffset(
                        state.src0, state.op, srcW, srcC, state.op.src0ElemSize);
                    const int64_t value = loadSignedValue(
                        src0Bytes, srcOffset, state.op.src0ElemSize);
                    accum += value;
                    currentMax = std::max(currentMax, value);
                }
                storeSignedValue(dstBytes, dstOffset,
                                 state.op.opcode == Opcode::VReduceSum ?
                                 accum : currentMax,
                                 state.op.dstElemSize);
            } else {
                uint64_t accum = 0;
                uint64_t currentMax = 0;
                bool first = true;
                const uint32_t reduceLimit = reduceAlongW ? state.src0.shape.w :
                    (reduceAlongC ? state.src0.shape.c :
                                    state.src0.shape.w * state.src0.shape.c);
                for (uint32_t idx = 0; idx < reduceLimit; ++idx) {
                    const uint32_t srcW = reduceAlongW ? idx :
                        (reduceAlongC ? w : (idx / state.src0.shape.c));
                    const uint32_t srcC = reduceAlongW ? c :
                        (reduceAlongC ? idx : (idx % state.src0.shape.c));
                    const size_t srcOffset = tensorElemOffset(
                        state.src0, state.op, srcW, srcC, state.op.src0ElemSize);
                    const uint64_t value = loadUnsignedValue(
                        src0Bytes, srcOffset, state.op.src0ElemSize);
                    accum += value;
                    currentMax = first ? value : std::max(currentMax, value);
                    first = false;
                }
                storeUnsignedValue(dstBytes, dstOffset,
                                   state.op.opcode == Opcode::VReduceSum ?
                                   accum : currentMax,
                                   state.op.dstElemSize);
            }
        }
    }

    if (state.dstInSpm) {
        state.resultBytes = std::move(dstBytes);
        state.resultReady = true;
    } else {
        writeResultBytes(state.dst, dstBytes, state.dstSpanBytes);
        state.resultReady = true;
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
    if (state.op.opcode == static_cast<Opcode>(VAbsOpcodeValue)) {
        executeUnary(state);
        return;
    }

    switch (state.op.opcode) {
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
        executeUnary(state);
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

    const TensorDesc tensor = kind == MacroCmdKind::Load ?
        decodeTensor(cmd, Src0AddrWord) : decodeTensor(cmd, DstAddrWord);
    return 1 + decodeSpmPort(tensor.addr, 1U);
}

bool
VpuUnit::canActivateMacroCmd(const MacroCmdContext &macroCmd) const
{
    (void)macroCmd;
    return true;
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
      case Opcode::VLoad:
        appendLoadUop(macroCmd, state.src0.addr, state.src0SpanBytes);
        macroCmd.uopQueue.back().portId =
            decodeSpmPort(state.src0.addr, state.src0SpanBytes);
        break;
      case Opcode::VStore: {
        if (!storeSourceReady(state)) {
            appendExecUop(macroCmd, 1);
            macroCmd.uopQueue.back().token =
                static_cast<uint64_t>(ExecToken::WaitStoreData);
            break;
        }
        const BufferRole role = isOutputLocalAddr(state.src0.addr) ?
            BufferRole::Output : BufferRole::Input;
        const LocalAddr src = decodeLocalAddr(state.src0.addr, role,
                                              state.src0SpanBytes);
        const auto &slot = bufferSlot(src);
        appendStoreUop(
            macroCmd, state.dst.addr, state.dstSpanBytes,
            std::vector<uint8_t>(slot.bytes.begin() + src.offset,
                                 slot.bytes.begin() + src.offset +
                                     state.src0SpanBytes));
        macroCmd.uopQueue.back().portId =
            decodeSpmPort(state.dst.addr, state.dstSpanBytes);
        break;
      }
      default:
        if (state.src0InSpm && !state.src0Loaded) {
            if (!state.src0Requested) {
                appendLoadUop(macroCmd, state.src0.addr, state.src0SpanBytes);
                macroCmd.uopQueue.back().token =
                    static_cast<uint64_t>(ExecToken::LoadSrc0);
                macroCmd.uopQueue.back().portId =
                    decodeSpmPort(state.src0.addr, state.src0SpanBytes);
                state.src0Requested = true;
            }
            break;
        }
        if (state.src1InSpm && !state.src1Loaded) {
            if (!state.src1Requested) {
                appendLoadUop(macroCmd, state.src1.addr, state.src1SpanBytes);
                macroCmd.uopQueue.back().token =
                    static_cast<uint64_t>(ExecToken::LoadSrc1);
                macroCmd.uopQueue.back().portId =
                    decodeSpmPort(state.src1.addr, state.src1SpanBytes);
                state.src1Requested = true;
            }
            break;
        }
        if (state.resultReady && state.dstInSpm && !state.storeIssued) {
            appendStoreUop(macroCmd, state.dst.addr, state.dstSpanBytes,
                           state.resultBytes);
            macroCmd.uopQueue.back().portId =
                decodeSpmPort(state.dst.addr, state.dstSpanBytes);
            state.storeIssued = true;
            break;
        }
        if (state.resultReady) {
            break;
        }
        appendExecUop(macroCmd, computeExecLatency(state));
        macroCmd.uopQueue.back().token =
            static_cast<uint64_t>(ExecToken::RunCompute);
        break;
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
        const LocalAddr dst = decodeLocalAddr(state.dst.addr, BufferRole::Input,
                                              state.dstSpanBytes);
        auto &slot = bufferSlot(dst);
        const size_t required = dst.offset + pkt->getSize();
        if (slot.bytes.size() < required) {
            slot.bytes.resize(required, 0);
        }
        std::copy(pkt->getConstPtr<uint8_t>(),
                  pkt->getConstPtr<uint8_t>() + pkt->getSize(),
                  slot.bytes.begin() + dst.offset);
        slot.valid = true;
        markEpiloguePending(macroCmd);
        return;
    }

    if (state.op.opcode != Opcode::VStore) {
        const ExecToken token = static_cast<ExecToken>(txn.token);
        if (token == ExecToken::LoadSrc0) {
            state.src0Bytes.assign(pkt->getConstPtr<uint8_t>(),
                                   pkt->getConstPtr<uint8_t>() + pkt->getSize());
            state.src0Loaded = true;
        } else if (token == ExecToken::LoadSrc1) {
            state.src1Bytes.assign(pkt->getConstPtr<uint8_t>(),
                                   pkt->getConstPtr<uint8_t>() + pkt->getSize());
            state.src1Loaded = true;
        }
        buildUops(macroCmd);
        if (macroCmd.uopQueue.empty() && state.resultReady &&
            (!state.dstInSpm || state.storeIssued)) {
            markEpiloguePending(macroCmd);
        }
        return;
    }

    markEpiloguePending(macroCmd);
}

void
VpuUnit::onExecUopComplete(MacroCmdContext &macroCmd,
                           const MicroOpContext &uop)
{
    auto it = macroStates.find(macroCmd.macroCmdId);
    panic_if(it == macroStates.end(), "%s: missing VPU macro state", name());
    auto &state = it->second;

    const ExecToken token = static_cast<ExecToken>(uop.token);
    if (state.op.opcode == Opcode::VStore || token == ExecToken::WaitStoreData) {
        buildUops(macroCmd);
        if (macroCmd.uopQueue.empty()) {
            markEpiloguePending(macroCmd);
        }
        return;
    }

    executeVectorOp(state);
    state.completedExecUops++;
    buildUops(macroCmd);
    if (macroCmd.uopQueue.empty() && (!state.dstInSpm || state.storeIssued)) {
        markEpiloguePending(macroCmd);
    }
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

const char *
VpuUnit::profileSeuType() const
{
    return "VPU";
}

void
VpuUnit::appendProfileDetailsJson(const MacroCmdContext &macroCmd,
                                  std::ostream &os) const
{
    const auto it = macroStates.find(macroCmd.macroCmdId);
    panic_if(it == macroStates.end(),
             "%s: missing VPU macro state for profiling", name());
    const auto &state = it->second;

    os << "\"opcode\":";
    appendJsonString(os, opcodeName(state.op.opcode));
    os << ",\"dst_type\":";
    appendJsonString(os, dataTypeName(state.op.dstType));
    os << ",\"src0_type\":";
    appendJsonString(os, dataTypeName(state.op.src0Type));
    os << ",\"src1_type\":";
    appendJsonString(os, dataTypeName(state.op.src1Type));
    os << ",\"w_layout_elems\":" << state.op.wLayoutElems;
    os << ",\"c_layout_elems\":" << state.op.cLayoutElems;
    os << ",\"scalar_bits\":" << state.op.scalarBits;
    os << ",\"dst_addr\":" << state.dst.addr;
    os << ",\"dst_shape_w\":" << state.dst.shape.w;
    os << ",\"dst_shape_c\":" << state.dst.shape.c;
    os << ",\"src0_addr\":" << state.src0.addr;
    os << ",\"src0_shape_w\":" << state.src0.shape.w;
    os << ",\"src0_shape_c\":" << state.src0.shape.c;
    os << ",\"src1_addr\":" << state.src1.addr;
    os << ",\"src1_shape_w\":" << state.src1.shape.w;
    os << ",\"src1_shape_c\":" << state.src1.shape.c;
    os << ",\"work_dlen_chunks\":" << workDlenChunks(state);
    os << ",\"dst_cycles_per_dlen\":" <<
        static_cast<uint64_t>(dtypeCyclesPerDlen(state.op.dstType));
    os << ",\"src0_cycles_per_dlen\":" <<
        static_cast<uint64_t>(dtypeCyclesPerDlen(state.op.src0Type));
    os << ",\"src1_cycles_per_dlen\":" <<
        (state.op.hasSrc1 ?
             static_cast<uint64_t>(dtypeCyclesPerDlen(state.op.src1Type)) : 0);
    os << ",\"completed_exec_uops\":" << state.completedExecUops;
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
