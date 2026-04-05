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

#ifndef __NPU_MEGA_VPU_UNIT_HH__
#define __NPU_MEGA_VPU_UNIT_HH__

#include <cstddef>
#include <cstdint>
#include <vector>

#include "npu/mega/LutUnit.hh"
#include "npu/mega/SpecializedExecutionUnit.hh"
#include "params/VpuUnit.hh"

namespace gem5
{

class VpuUnit : public SpecializedExecutionUnit
{
  private:
    static constexpr uint8_t VpuDeviceType = 0x2;
    static constexpr Addr SpmBase = 0x60000000;
    static constexpr Addr SpmSlotStride = 0x40;
    static constexpr size_t FlagsWord = 4;
    static constexpr size_t ElemCountWord = 5;
    static constexpr size_t SrcStrideWord = 6;
    static constexpr size_t DstStrideWord = 7;
    static constexpr size_t DataTypeWord = 8;
    static constexpr size_t ScalarBitsWord = 9;

    enum class Opcode : uint8_t
    {
        Exec = 0x0,
        VAdd = 0x1,
        VSub = 0x2,
        VMul = 0x3,
        VDiv = 0x4,
        VScale = 0x5,
        VCvtI2F = 0x6,
        VCvtF2I = 0x7,
        VSqrt = 0x8,
        VFma = 0x9,
        VReduceSum = 0xa,
        VReduceMax = 0xb,
        VLoad = 0xc,
        VStore = 0xd,
        VExp = 0xe,
        VSoftmax = 0xf,
    };

    enum class DataType : uint32_t
    {
        Int32 = 0x0,
        Float32 = 0x1,
    };

    struct DecodedVectorOp
    {
        Opcode opcode = Opcode::Exec;
        DataType dataType = DataType::Int32;
        uint32_t flags = 0;
        uint32_t elemCount = 0;
        uint32_t srcStrideBytes = 0;
        uint32_t dstStrideBytes = 0;
        uint32_t scalarBits = 0;
        size_t elemSize = sizeof(uint32_t);
        size_t srcSpanBytes = sizeof(uint32_t);
        size_t dstSpanBytes = sizeof(uint32_t);
        bool legacyExec = true;
    };

    const uint8_t deviceId;
    LutUnit *const lut;
    std::vector<std::vector<uint8_t>> residentBuffers;
    std::vector<bool> residentBufferValid;
    Tick lastLinearExecuteLatencyValue = 0;
    Tick lastLinearCompletionTickValue = 0;

    uint32_t unpackWord(const std::vector<uint8_t> &bytes) const;
    std::vector<uint8_t> packWord(uint32_t value) const;
    uint32_t cmdWord(const std::vector<uint8_t> &cmd, size_t wordIdx) const;
    Addr slotAddr(PortID portId, Addr offset = 0) const;
    Opcode decodeOpcode(uint8_t opCode) const;
    DecodedVectorOp decodeVectorOp(const ActiveExecution &exec) const;
    void validateVectorPortLayout(const ActiveExecution &exec,
                                  const DecodedVectorOp &op) const;
    uint32_t loadUint32(const std::vector<uint8_t> &bytes, size_t offset) const;
    float loadFloat32(const std::vector<uint8_t> &bytes, size_t offset) const;
    void storeUint32(std::vector<uint8_t> &bytes, size_t offset,
                     uint32_t value) const;
    void storeFloat32(std::vector<uint8_t> &bytes, size_t offset,
                      float value) const;
    bool isLutOpcode(Opcode opcode) const;
    bool isLinearOpcode(Opcode opcode) const;
    LutUnit::Operation lutOperation(Opcode opcode) const;
    void validateCommand(const ActiveExecution &exec) const;

  protected:
    void onCommandBegin(ActiveExecution &exec) override;
    void buildMvinRequests(ActiveExecution &exec,
                           std::vector<MemRequestDesc> &reqs) override;
    void onMvinResponse(ActiveExecution &exec, const MemTxnContext &txn,
                        PacketPtr pkt) override;
    Tick execute(ActiveExecution &exec) override;
    void buildMvoutRequests(ActiveExecution &exec,
                            std::vector<MemRequestDesc> &reqs) override;
    void onMvoutResponse(ActiveExecution &exec, const MemTxnContext &txn,
                         PacketPtr pkt) override;
    void epilogue(ActiveExecution &exec) override;

  public:
    VpuUnit(const VpuUnitParams &params);
    uint64_t lutRequestCount() const;
    uint64_t lutCommandCount() const;
    Tick lastLinearExecuteLatency() const;
    Tick lastLutExecuteLatency() const;
    Tick lastSoftmaxExecuteLatency() const;
    Tick lastLinearCompletionTick() const;
    Tick lastLutCompletionTick() const;
    Tick lastSoftmaxCompletionTick() const;
};

} // namespace gem5

#endif // __NPU_MEGA_VPU_UNIT_HH__
