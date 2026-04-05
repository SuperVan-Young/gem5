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

#include "npu/mega/VpuUnit.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/VPU.hh"

namespace gem5
{

VpuUnit::VpuUnit(const VpuUnitParams &params)
    : SpecializedExecutionUnit(params), deviceId(params.device_id),
      lut(params.lut),
      residentBuffers(params.num_mem_side_ports),
      residentBufferValid(params.num_mem_side_ports, false)
{
    panic_if(lut == nullptr, "%s: lut must not be null", name());
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
    const size_t offset = wordIdx * sizeof(uint32_t);
    if (cmd.size() < offset + sizeof(uint32_t)) {
        return 0;
    }

    uint32_t word = 0;
    std::memcpy(&word, cmd.data() + offset, sizeof(word));
    return word;
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

    panic("%s: unreachable VPU opcode decode", name());
}

VpuUnit::DecodedVectorOp
VpuUnit::decodeVectorOp(const ActiveExecution &exec) const
{
    DecodedVectorOp op;
    op.opcode = decodeOpcode(exec.fields.opCode);
    op.legacyExec = op.opcode == Opcode::Exec;
    if (op.legacyExec) {
        return op;
    }

    op.flags = cmdWord(exec.cmd, FlagsWord);
    op.elemCount = cmdWord(exec.cmd, ElemCountWord);
    op.srcStrideBytes = cmdWord(exec.cmd, SrcStrideWord);
    op.dstStrideBytes = cmdWord(exec.cmd, DstStrideWord);
    op.scalarBits = cmdWord(exec.cmd, ScalarBitsWord);

    const uint32_t rawDataType = cmdWord(exec.cmd, DataTypeWord);
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

    op.elemSize = sizeof(uint32_t);

    if (op.srcStrideBytes == 0) {
        op.srcStrideBytes = op.elemSize;
    }
    if (op.dstStrideBytes == 0) {
        op.dstStrideBytes = op.elemSize;
    }

    panic_if(op.srcStrideBytes < op.elemSize,
             "%s: src stride %u smaller than element size %zu",
             name(), op.srcStrideBytes, op.elemSize);
    panic_if(op.dstStrideBytes < op.elemSize,
             "%s: dst stride %u smaller than element size %zu",
             name(), op.dstStrideBytes, op.elemSize);

    op.srcSpanBytes = (static_cast<size_t>(op.elemCount - 1) *
                       static_cast<size_t>(op.srcStrideBytes)) + op.elemSize;
    op.dstSpanBytes = (static_cast<size_t>(op.elemCount - 1) *
                       static_cast<size_t>(op.dstStrideBytes)) + op.elemSize;

    panic_if(op.srcSpanBytes > SpmSlotStride,
             "%s: source span %zu exceeds SPM slot stride %#llx",
             name(), op.srcSpanBytes,
             static_cast<unsigned long long>(SpmSlotStride));
    panic_if(op.dstSpanBytes > SpmSlotStride,
             "%s: destination span %zu exceeds SPM slot stride %#llx",
             name(), op.dstSpanBytes,
             static_cast<unsigned long long>(SpmSlotStride));
    panic_if(op.flags != 0,
             "%s: unsupported VPU flags=%#x for opCode=%u", name(), op.flags,
             exec.fields.opCode);

    if (op.opcode == Opcode::VCvtI2F) {
        panic_if(op.dataType != DataType::Float32,
                 "%s: VCvtI2F requires Float32 destination data type",
                 name());
    }
    if (op.opcode == Opcode::VCvtF2I) {
        panic_if(op.dataType != DataType::Int32,
                 "%s: VCvtF2I requires Int32 destination data type", name());
    }
    if (op.opcode == Opcode::VSqrt) {
        panic_if(op.dataType != DataType::Float32,
                 "%s: VSqrt requires Float32 data type", name());
    }
    if (op.opcode == Opcode::VFma) {
        panic_if(op.dataType != DataType::Float32,
                 "%s: VFMA requires Float32 data type", name());
    }
    if (op.opcode == Opcode::VExp || op.opcode == Opcode::VSoftmax) {
        panic_if(op.dataType != DataType::Float32,
                 "%s: nonlinear LUT ops require Float32 data type", name());
    }

    return op;
}

void
VpuUnit::validateVectorPortLayout(const ActiveExecution &exec,
                                  const DecodedVectorOp &op) const
{
    if (op.legacyExec) {
        return;
    }

    switch (op.opcode) {
      case Opcode::VAdd:
      case Opcode::VSub:
      case Opcode::VMul:
      case Opcode::VDiv:
        panic_if(exec.writePorts.empty(),
                 "%s: vector op requires at least one write port", name());
        panic_if(exec.readPorts.size() != exec.writePorts.size() * 2,
                 "%s: binary vector op requires exactly two read ports per "
                 "write port (reads=%zu writes=%zu)",
                 name(), exec.readPorts.size(), exec.writePorts.size());
        break;
      case Opcode::VFma:
        panic_if(exec.writePorts.empty(),
                 "%s: vector op requires at least one write port", name());
        panic_if(exec.readPorts.size() != exec.writePorts.size() * 3,
                 "%s: VFMA requires exactly three read ports per write port "
                 "(reads=%zu writes=%zu)",
                 name(), exec.readPorts.size(), exec.writePorts.size());
        break;
      case Opcode::VReduceSum:
      case Opcode::VReduceMax:
        panic_if(exec.writePorts.empty(),
                 "%s: vector op requires at least one write port", name());
        panic_if(exec.readPorts.size() != exec.writePorts.size(),
                 "%s: reduce op requires one read port per write port "
                 "(reads=%zu writes=%zu)",
                 name(), exec.readPorts.size(), exec.writePorts.size());
        break;
      case Opcode::VLoad:
        panic_if(exec.readPorts.empty(),
                 "%s: VLOAD requires at least one read port", name());
        panic_if(!exec.writePorts.empty(),
                 "%s: VLOAD does not accept write ports", name());
        break;
      case Opcode::VStore:
        panic_if(!exec.readPorts.empty(),
                 "%s: VSTORE does not accept read ports", name());
        panic_if(exec.writePorts.empty(),
                 "%s: VSTORE requires at least one write port", name());
        break;
      case Opcode::VScale:
      case Opcode::VCvtI2F:
      case Opcode::VCvtF2I:
      case Opcode::VSqrt:
      case Opcode::VExp:
      case Opcode::VSoftmax:
        panic_if(exec.writePorts.empty(),
                 "%s: vector op requires at least one write port", name());
        panic_if(exec.readPorts.size() != exec.writePorts.size(),
                 "%s: unary vector op requires one read port per write port "
                 "(reads=%zu writes=%zu)",
                 name(), exec.readPorts.size(), exec.writePorts.size());
        break;
      case Opcode::Exec:
        panic("%s: unexpected legacy opcode in vector port validation",
              name());
    }
}

uint32_t
VpuUnit::loadUint32(const std::vector<uint8_t> &bytes, size_t offset) const
{
    panic_if(offset + sizeof(uint32_t) > bytes.size(),
             "%s: uint32 load out of range offset=%zu size=%zu",
             name(), offset, bytes.size());
    uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

float
VpuUnit::loadFloat32(const std::vector<uint8_t> &bytes, size_t offset) const
{
    float value = 0.0f;
    panic_if(offset + sizeof(value) > bytes.size(),
             "%s: float load out of range offset=%zu size=%zu",
             name(), offset, bytes.size());
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void
VpuUnit::storeUint32(std::vector<uint8_t> &bytes, size_t offset,
                     uint32_t value) const
{
    panic_if(offset + sizeof(value) > bytes.size(),
             "%s: uint32 store out of range offset=%zu size=%zu",
             name(), offset, bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void
VpuUnit::storeFloat32(std::vector<uint8_t> &bytes, size_t offset,
                      float value) const
{
    panic_if(offset + sizeof(value) > bytes.size(),
             "%s: float store out of range offset=%zu size=%zu",
             name(), offset, bytes.size());
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
    return !isLutOpcode(opcode) && opcode != Opcode::Exec;
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
        panic("%s: opcode %u does not use LUT", name(),
              static_cast<uint32_t>(opcode));
    }
}

void
VpuUnit::validateCommand(const ActiveExecution &exec) const
{
    fatal_if(exec.fields.deviceType != VpuDeviceType,
             "%s: unexpected deviceType=%u for VPU command", name(),
             exec.fields.deviceType);
    fatal_if(exec.fields.deviceId != deviceId,
             "%s: command deviceId=%u does not match instance deviceId=%u",
             name(), exec.fields.deviceId, deviceId);
    decodeVectorOp(exec);
}

void
VpuUnit::onCommandBegin(ActiveExecution &exec)
{
    validateCommand(exec);
    const DecodedVectorOp op = decodeVectorOp(exec);
    DPRINTF(VPU,
            "%s begin cmd deviceType=%u deviceId=%u opCode=%u sync=%u "
            "readMask=%#x writeMask=%#x repetition=%u elemCount=%u "
            "srcStride=%u dstStride=%u dataType=%u scalar=%#x flags=%#x\n",
            name(), exec.fields.deviceType, exec.fields.deviceId,
            exec.fields.opCode, exec.fields.syncIndicator, exec.readMask,
            exec.writeMask, exec.repetition, op.elemCount, op.srcStrideBytes,
            op.dstStrideBytes, static_cast<uint32_t>(op.dataType),
            op.scalarBits, op.flags);
}

void
VpuUnit::buildMvinRequests(ActiveExecution &exec,
                           std::vector<MemRequestDesc> &reqs)
{
    const DecodedVectorOp op = decodeVectorOp(exec);
    validateVectorPortLayout(exec, op);

    if (op.opcode == Opcode::VStore) {
        return;
    }

    for (const PortID port : exec.readPorts) {
        MemRequestDesc req;
        req.portId = port;
        req.kind = MemTxnContext::Kind::Mvin;
        req.addr = slotAddr(port);
        req.size = op.legacyExec ? sizeof(uint32_t) : op.srcSpanBytes;
        reqs.push_back(req);

        DPRINTF(VPU,
                "%s iteration=%llu issue read port=%d addr=%#llx size=%u\n",
                name(),
                static_cast<unsigned long long>(exec.iteration),
                port, static_cast<unsigned long long>(req.addr),
                static_cast<unsigned>(req.size));
    }
}

void
VpuUnit::onMvinResponse(ActiveExecution &exec, const MemTxnContext &txn,
                        PacketPtr pkt)
{
    uint32_t value = 0;
    const size_t copy_size =
        std::min(static_cast<size_t>(pkt->getSize()), sizeof(value));
    std::memcpy(&value, pkt->getConstPtr<uint8_t>(),
                copy_size);
    DPRINTF(VPU,
            "%s iteration=%llu read response port=%d addr=%#llx value=%#x\n",
            name(),
            static_cast<unsigned long long>(exec.iteration), txn.portId,
            static_cast<unsigned long long>(txn.addr), value);
}

Tick
VpuUnit::execute(ActiveExecution &exec)
{
    const DecodedVectorOp op = decodeVectorOp(exec);
    Tick extraLatency = 0;

    if (isLutOpcode(op.opcode)) {
        const uint32_t lutRequests =
            op.opcode == Opcode::VSoftmax ?
            (op.elemCount * exec.writePorts.size()) :
            (op.elemCount * exec.writePorts.size());
        extraLatency = lut->reserve(lutOperation(op.opcode), lutRequests,
                                    curTick());
    }

    if (!op.legacyExec) {
        if (op.opcode == Opcode::VLoad) {
            for (const PortID port : exec.readPorts) {
                panic_if(port < 0 ||
                         static_cast<size_t>(port) >= residentBuffers.size(),
                         "%s: VLOAD port %d out of resident buffer range",
                         name(), port);
                residentBuffers[port] = exec.readResults[port];
                residentBufferValid[port] = true;
                DPRINTF(VPU,
                        "%s iteration=%llu execute vload port=%d bytes=%u\n",
                        name(),
                        static_cast<unsigned long long>(exec.iteration), port,
                        static_cast<unsigned>(residentBuffers[port].size()));
            }
        }

        if (op.opcode == Opcode::VStore) {
            for (const PortID port : exec.writePorts) {
                panic_if(port < 0 ||
                         static_cast<size_t>(port) >= residentBuffers.size(),
                         "%s: VSTORE port %d out of resident buffer range",
                         name(), port);
                panic_if(!residentBufferValid[port],
                         "%s: VSTORE port %d has no loaded resident buffer",
                         name(), port);
                panic_if(residentBuffers[port].size() < op.dstSpanBytes,
                         "%s: VSTORE port %d resident buffer too small "
                         "(have=%zu need=%zu)",
                         name(), port, residentBuffers[port].size(),
                         op.dstSpanBytes);

                exec.writeResults[port] = std::vector<uint8_t>(
                    residentBuffers[port].begin(),
                    residentBuffers[port].begin() + op.dstSpanBytes);
                DPRINTF(VPU,
                        "%s iteration=%llu execute vstore port=%d bytes=%u\n",
                        name(),
                        static_cast<unsigned long long>(exec.iteration), port,
                        static_cast<unsigned>(op.dstSpanBytes));
            }
        }

        for (size_t dstIndex = 0; dstIndex < exec.writePorts.size(); ++dstIndex) {
            const PortID dstPort = exec.writePorts[dstIndex];
            std::vector<uint8_t> dstBytes(op.dstSpanBytes, 0);

            if (op.opcode == Opcode::VAdd || op.opcode == Opcode::VSub ||
                op.opcode == Opcode::VMul || op.opcode == Opcode::VDiv) {
                const PortID lhsPort = exec.readPorts[dstIndex * 2];
                const PortID rhsPort = exec.readPorts[(dstIndex * 2) + 1];
                const auto &lhsBytes = exec.readResults[lhsPort];
                const auto &rhsBytes = exec.readResults[rhsPort];

                for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                    const size_t srcOffset =
                        static_cast<size_t>(elem) * op.srcStrideBytes;
                    const size_t dstOffset =
                        static_cast<size_t>(elem) * op.dstStrideBytes;

                    if (op.dataType == DataType::Int32) {
                        const uint32_t lhs = loadUint32(lhsBytes, srcOffset);
                        const uint32_t rhs = loadUint32(rhsBytes, srcOffset);
                        uint32_t value = 0;
                        switch (op.opcode) {
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
                            const int32_t lhsSigned =
                                static_cast<int32_t>(lhs);
                            const int32_t rhsSigned =
                                static_cast<int32_t>(rhs);
                            int32_t quotient = 0;
                            if (rhsSigned == 0) {
                                quotient = 0;
                            } else if (
                                lhsSigned == std::numeric_limits<int32_t>::min()
                                && rhsSigned == -1) {
                                quotient =
                                    std::numeric_limits<int32_t>::max();
                            } else {
                                quotient = lhsSigned / rhsSigned;
                            }
                            value = static_cast<uint32_t>(quotient);
                            break;
                          }
                          default:
                            panic("%s: unexpected opcode in binary int path",
                                  name());
                        }
                        storeUint32(dstBytes, dstOffset, value);
                    } else {
                        const float lhs = loadFloat32(lhsBytes, srcOffset);
                        const float rhs = loadFloat32(rhsBytes, srcOffset);
                        float value = 0.0f;
                        switch (op.opcode) {
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
                            panic("%s: unexpected opcode in binary float path",
                                  name());
                        }
                        storeFloat32(dstBytes, dstOffset, value);
                    }
                }

                exec.writeResults[dstPort] = dstBytes;
                DPRINTF(VPU,
                        "%s iteration=%llu execute binary op=%u dstPort=%d "
                        "lhsPort=%d rhsPort=%d elemCount=%u\n",
                        name(), static_cast<unsigned long long>(exec.iteration),
                        exec.fields.opCode, dstPort, lhsPort, rhsPort,
                        op.elemCount);
                continue;
            }

            if (op.opcode == Opcode::VFma) {
                const PortID src0Port = exec.readPorts[dstIndex * 3];
                const PortID src1Port = exec.readPorts[(dstIndex * 3) + 1];
                const PortID src2Port = exec.readPorts[(dstIndex * 3) + 2];
                const auto &src0Bytes = exec.readResults[src0Port];
                const auto &src1Bytes = exec.readResults[src1Port];
                const auto &src2Bytes = exec.readResults[src2Port];

                for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                    const size_t srcOffset =
                        static_cast<size_t>(elem) * op.srcStrideBytes;
                    const size_t dstOffset =
                        static_cast<size_t>(elem) * op.dstStrideBytes;
                    const float src0 = loadFloat32(src0Bytes, srcOffset);
                    const float src1 = loadFloat32(src1Bytes, srcOffset);
                    const float src2 = loadFloat32(src2Bytes, srcOffset);
                    const float value = std::fma(src0, src1, src2);
                    storeFloat32(dstBytes, dstOffset, value);
                }

                exec.writeResults[dstPort] = dstBytes;
                DPRINTF(VPU,
                        "%s iteration=%llu execute vfma dstPort=%d "
                        "src0Port=%d src1Port=%d src2Port=%d elemCount=%u\n",
                        name(), static_cast<unsigned long long>(exec.iteration),
                        dstPort, src0Port, src1Port, src2Port, op.elemCount);
                continue;
            }

            if (op.opcode == Opcode::VReduceSum ||
                op.opcode == Opcode::VReduceMax) {
                const PortID srcPort = exec.readPorts[dstIndex];
                const auto &srcBytes = exec.readResults[srcPort];
                std::vector<uint8_t> reduceBytes(sizeof(uint32_t), 0);

                if (op.dataType == DataType::Int32) {
                    int32_t accum = 0;
                    int32_t current_max = std::numeric_limits<int32_t>::min();
                    for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                        const size_t srcOffset =
                            static_cast<size_t>(elem) * op.srcStrideBytes;
                        const int32_t value = static_cast<int32_t>(
                            loadUint32(srcBytes, srcOffset));
                        accum += value;
                        current_max = std::max(current_max, value);
                    }

                    const int32_t result =
                        op.opcode == Opcode::VReduceSum ? accum : current_max;
                    storeUint32(reduceBytes, 0,
                                static_cast<uint32_t>(result));
                } else {
                    float accum = 0.0f;
                    float current_max = -std::numeric_limits<float>::infinity();
                    for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                        const size_t srcOffset =
                            static_cast<size_t>(elem) * op.srcStrideBytes;
                        const float value = loadFloat32(srcBytes, srcOffset);
                        accum += value;
                        current_max = std::max(current_max, value);
                    }

                    const float result =
                        op.opcode == Opcode::VReduceSum ? accum : current_max;
                    storeFloat32(reduceBytes, 0, result);
                }

                exec.writeResults[dstPort] = reduceBytes;
                DPRINTF(VPU,
                        "%s iteration=%llu execute reduce op=%u srcPort=%d "
                        "dstPort=%d elemCount=%u\n",
                        name(), static_cast<unsigned long long>(exec.iteration),
                        exec.fields.opCode, srcPort, dstPort, op.elemCount);
                continue;
            }

            if (op.opcode == Opcode::VLoad || op.opcode == Opcode::VStore) {
                continue;
            }

            const PortID srcPort = exec.readPorts[dstIndex];
            const auto &srcBytes = exec.readResults[srcPort];

            if (op.opcode == Opcode::VSoftmax) {
                const PortID srcPort = exec.readPorts[dstIndex];
                const auto &srcBytes = exec.readResults[srcPort];
                std::vector<float> expValues(op.elemCount, 0.0f);
                float maxValue = -std::numeric_limits<float>::infinity();
                float sum = 0.0f;

                for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                    const size_t srcOffset =
                        static_cast<size_t>(elem) * op.srcStrideBytes;
                    maxValue = std::max(maxValue,
                                        loadFloat32(srcBytes, srcOffset));
                }

                for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                    const size_t srcOffset =
                        static_cast<size_t>(elem) * op.srcStrideBytes;
                    const float shifted =
                        loadFloat32(srcBytes, srcOffset) - maxValue;
                    expValues[elem] = lut->evaluateExp(shifted);
                    sum += expValues[elem];
                }

                for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                    const size_t dstOffset =
                        static_cast<size_t>(elem) * op.dstStrideBytes;
                    storeFloat32(dstBytes, dstOffset, expValues[elem] / sum);
                }
            } else {
                for (uint32_t elem = 0; elem < op.elemCount; ++elem) {
                    const size_t srcOffset =
                        static_cast<size_t>(elem) * op.srcStrideBytes;
                    const size_t dstOffset =
                        static_cast<size_t>(elem) * op.dstStrideBytes;

                    switch (op.opcode) {
                      case Opcode::VScale:
                        if (op.dataType == DataType::Int32) {
                            const int32_t value = static_cast<int32_t>(
                                loadUint32(srcBytes, srcOffset));
                            const int32_t scalar = static_cast<int32_t>(
                                op.scalarBits);
                            const int32_t scaled = value * scalar;
                            storeUint32(dstBytes, dstOffset,
                                        static_cast<uint32_t>(scaled));
                        } else {
                            const float value = loadFloat32(srcBytes,
                                                            srcOffset);
                            float scalar = 0.0f;
                            std::memcpy(&scalar, &op.scalarBits,
                                        sizeof(scalar));
                            storeFloat32(dstBytes, dstOffset, value * scalar);
                        }
                        break;
                      case Opcode::VCvtI2F: {
                        const int32_t value = static_cast<int32_t>(
                            loadUint32(srcBytes, srcOffset));
                        storeFloat32(dstBytes, dstOffset,
                                     static_cast<float>(value));
                        break;
                      }
                      case Opcode::VCvtF2I: {
                        const float value = loadFloat32(srcBytes, srcOffset);
                        int32_t converted = 0;
                        if (std::isnan(value)) {
                            converted = 0;
                        } else if (value >=
                                   static_cast<float>(
                                       std::numeric_limits<int32_t>::max())) {
                            converted = std::numeric_limits<int32_t>::max();
                        } else if (value <=
                                   static_cast<float>(
                                       std::numeric_limits<int32_t>::min())) {
                            converted = std::numeric_limits<int32_t>::min();
                        } else {
                            converted = static_cast<int32_t>(std::trunc(
                                value));
                        }
                        storeUint32(dstBytes, dstOffset,
                                    static_cast<uint32_t>(converted));
                        break;
                      }
                      case Opcode::VSqrt: {
                        const float value = loadFloat32(srcBytes, srcOffset);
                        storeFloat32(dstBytes, dstOffset,
                                     lut->evaluateSqrt(value));
                        break;
                      }
                      case Opcode::VExp: {
                        const float value = loadFloat32(srcBytes, srcOffset);
                        storeFloat32(dstBytes, dstOffset,
                                     lut->evaluateExp(value));
                        break;
                      }
                      default:
                        panic("%s: unexpected opcode in unary path", name());
                    }
                }
            }

            exec.writeResults[dstPort] = dstBytes;
            DPRINTF(VPU,
                    "%s iteration=%llu execute unary op=%u srcPort=%d "
                    "dstPort=%d elemCount=%u scalar=%#x\n",
                    name(), static_cast<unsigned long long>(exec.iteration),
                    exec.fields.opCode, srcPort, dstPort, op.elemCount,
                    op.scalarBits);
        }
    }

    DPRINTF(VPU,
            "%s iteration=%llu execute readPorts=%u writePorts=%u\n",
            name(), static_cast<unsigned long long>(exec.iteration),
            static_cast<unsigned>(exec.readPorts.size()),
            static_cast<unsigned>(exec.writePorts.size()));
    const Tick baseLatency = SpecializedExecutionUnit::execute(exec);
    const Tick totalLatency = baseLatency + extraLatency;

    if (isLinearOpcode(op.opcode)) {
        lastLinearExecuteLatencyValue = totalLatency;
    }

    return totalLatency;
}

void
VpuUnit::buildMvoutRequests(ActiveExecution &exec,
                            std::vector<MemRequestDesc> &reqs)
{
    const DecodedVectorOp op = decodeVectorOp(exec);
    if (!op.legacyExec) {
        if (op.opcode == Opcode::VLoad) {
            return;
        }

        for (const PortID port : exec.writePorts) {
            auto resultIt = exec.writeResults.find(port);
            panic_if(resultIt == exec.writeResults.end(),
                     "%s: missing vector write result for port=%d",
                     name(), port);

            MemRequestDesc req;
            req.portId = port;
            req.kind = MemTxnContext::Kind::Mvout;
            req.addr = slotAddr(port);
            req.size = (op.opcode == Opcode::VReduceSum ||
                        op.opcode == Opcode::VReduceMax) ?
                sizeof(uint32_t) : op.dstSpanBytes;
            req.data = resultIt->second;
            reqs.push_back(req);

            DPRINTF(VPU,
                    "%s iteration=%llu issue vector write port=%d addr=%#llx "
                    "size=%u elemCount=%u\n",
                    name(),
                    static_cast<unsigned long long>(exec.iteration), port,
                    static_cast<unsigned long long>(req.addr),
                    static_cast<unsigned>(req.size),
                    op.elemCount);
        }
        return;
    }

    uint32_t signature = 0;
    for (const PortID port : exec.readPorts) {
        signature += unpackWord(exec.readResults[port]);
    }

    for (const PortID port : exec.writePorts) {
        const uint32_t current = unpackWord(exec.readResults[port]);
        const uint32_t value = current + signature +
            ((static_cast<uint32_t>(exec.iteration) + 1U) * 0x10U) +
            (static_cast<uint32_t>(port) + 1U);
        exec.writeResults[port] = packWord(value);

        MemRequestDesc req;
        req.portId = port;
        req.kind = MemTxnContext::Kind::Mvout;
        req.addr = slotAddr(port);
        req.size = sizeof(uint32_t);
        req.data = exec.writeResults[port];
        reqs.push_back(req);

        DPRINTF(VPU,
                "%s iteration=%llu issue write port=%d addr=%#llx value=%#x "
                "signature=%#x\n",
                name(),
                static_cast<unsigned long long>(exec.iteration), port,
                static_cast<unsigned long long>(req.addr), value, signature);
    }
}

void
VpuUnit::onMvoutResponse(ActiveExecution &exec, const MemTxnContext &txn,
                         PacketPtr pkt)
{
    const uint32_t value = unpackWord(exec.writeResults[txn.portId]);
    DPRINTF(VPU,
            "%s iteration=%llu write response port=%d addr=%#llx value=%#x "
            "size=%u\n",
            name(),
            static_cast<unsigned long long>(exec.iteration), txn.portId,
            static_cast<unsigned long long>(txn.addr), value, pkt->getSize());
}

void
VpuUnit::epilogue(ActiveExecution &exec)
{
    const Opcode opcode = decodeVectorOp(exec).opcode;
    if (isLutOpcode(opcode)) {
        lut->noteCompletion(lutOperation(opcode), curTick());
    } else if (isLinearOpcode(opcode)) {
        lastLinearCompletionTickValue = curTick();
    }

    DPRINTF(VPU,
            "%s iteration=%llu complete completedIterations=%llu "
            "completedReads=%llu completedWrites=%llu\n",
            name(), static_cast<unsigned long long>(exec.iteration),
            static_cast<unsigned long long>(exec.completedIterations),
            static_cast<unsigned long long>(exec.completedReadRespCount),
            static_cast<unsigned long long>(exec.completedWriteRespCount));
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
