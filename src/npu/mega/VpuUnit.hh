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
#include <unordered_map>
#include <vector>

#include "npu/mega/LutUnit.hh"
#include "npu/mega/SpecializedExecutionUnit.hh"
#include "params/VpuUnit.hh"

namespace gem5
{

/*
 * VpuUnit is now a semantic layer on top of the generic SEU issue/uop
 * infrastructure.
 *
 * Public infrastructure inherited from SpecializedExecutionUnit:
 * - dispatch/issue scheduling
 * - mem-side transaction routing
 * - exec uop timing and callback delivery
 * - completion sync writes and queue ownership
 *
 * VpuUnit-specific responsibilities:
 * - decode VPU commands
 * - classify commands into load/store/compute
 * - validate SPM-vs-local-buffer address usage
 * - manage the configurable local input/output buffer arrays
 * - run vector arithmetic in exec callbacks
 *
 * Current modeling rule:
 * - load/store commands may access SPM
 * - compute commands must use local buffer addresses only
 * - compute commands only schedule exec uops in this revision
 * - helper code may decompose one software-level VPU operation into several
 *   load/compute/store macro commands
 */
class VpuUnit : public SpecializedExecutionUnit
{
  private:
    static constexpr uint8_t VpuDeviceType = 0x2;
    static constexpr Addr SpmBase = 0x60000000;
    static constexpr Addr SpmSlotStride = 0x40;

    static constexpr size_t ReadMaskWord = 1;
    static constexpr size_t WriteMaskWord = 2;
    static constexpr size_t RepetitionWord = 3;
    static constexpr size_t FlagsWord = 4;
    static constexpr size_t ElemCountWord = 5;
    static constexpr size_t SrcStrideWord = 6;
    static constexpr size_t DstStrideWord = 7;
    static constexpr size_t DataTypeWord = 8;
    static constexpr size_t ScalarBitsWord = 9;
    static constexpr size_t Src0AddrWord = 10;
    static constexpr size_t Src1AddrWord = 11;
    static constexpr size_t Src2AddrWord = 12;
    static constexpr size_t DstAddrWord = 13;

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

    enum class BufferRole : uint8_t
    {
        Input,
        Output,
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
        uint32_t repetition = 1;
        size_t elemSize = sizeof(uint32_t);
        size_t srcSpanBytes = sizeof(uint32_t);
        size_t dstSpanBytes = sizeof(uint32_t);
        bool legacyExec = false;
    };

    struct LocalAddr
    {
        BufferRole role = BufferRole::Input;
        PortID portId = InvalidPortID;
        uint32_t bufferIndex = 0;
        size_t offset = 0;
    };

    struct LocalBufferSlot
    {
        std::vector<uint8_t> bytes;
        bool valid = false;
    };

    struct VpuMacroState
    {
        DecodedVectorOp op;
        uint32_t readMask = 0;
        uint32_t writeMask = 0;
        Addr src0Addr = 0;
        Addr src1Addr = 0;
        Addr src2Addr = 0;
        Addr dstAddr = 0;
        std::vector<PortID> readPorts;
        std::vector<PortID> writePorts;
        size_t completedExecUops = 0;
    };

    enum class ExecToken : uint64_t
    {
        WaitInputs = 1,
        RunCompute = 2,
        WaitStoreData = 3,
    };

    const uint8_t deviceId;
    LutUnit *const lut;
    const uint32_t numMemPorts;
    const uint32_t numInputPorts;
    const uint32_t numOutputPorts;
    const uint32_t inputBufferCount;
    const uint32_t outputBufferCount;
    const Addr localInputBase;
    const Addr localOutputBase;
    const uint32_t localBufferStride;

    std::vector<std::vector<LocalBufferSlot>> inputBuffers;
    std::vector<std::vector<LocalBufferSlot>> outputBuffers;
    std::unordered_map<uint64_t, VpuMacroState> macroStates;

    Tick lastLinearExecuteLatencyValue = 0;
    Tick lastLinearCompletionTickValue = 0;

    uint32_t unpackWord(const std::vector<uint8_t> &bytes) const;
    std::vector<uint8_t> packWord(uint32_t value) const;
    uint32_t cmdWord(const std::vector<uint8_t> &cmd, size_t wordIdx) const;
    Addr slotAddr(PortID portId, Addr offset = 0) const;
    Opcode decodeOpcode(uint8_t opCode) const;
    DecodedVectorOp decodeVectorOp(const MacroCmdContext &macroCmd) const;
    std::vector<PortID> decodeMask(uint32_t mask) const;
    void validatePortLayout(const VpuMacroState &state) const;
    void validateCommand(const MacroCmdContext &macroCmd,
                         VpuMacroState &state) const;
    bool isSpmAddr(Addr addr) const;
    bool isInputLocalAddr(Addr addr) const;
    bool isOutputLocalAddr(Addr addr) const;
    LocalAddr decodeLocalAddr(Addr addr, BufferRole expected,
                              size_t accessSize) const;
    PortID decodeSpmPort(Addr addr, size_t accessSize) const;
    Tick computeExecLatency(const VpuMacroState &state);
    uint32_t loadUint32(
        const std::vector<uint8_t> &bytes, size_t offset) const;
    float loadFloat32(const std::vector<uint8_t> &bytes, size_t offset) const;
    void storeUint32(std::vector<uint8_t> &bytes, size_t offset,
                     uint32_t value) const;
    void storeFloat32(std::vector<uint8_t> &bytes, size_t offset,
                      float value) const;
    bool isLutOpcode(Opcode opcode) const;
    bool isLinearOpcode(Opcode opcode) const;
    LutUnit::Operation lutOperation(Opcode opcode) const;
    LocalBufferSlot &bufferSlot(const LocalAddr &addr);
    const LocalBufferSlot &bufferSlot(const LocalAddr &addr) const;
    const std::vector<uint8_t> &sourceBytes(const VpuMacroState &state,
                                            size_t sourceIndex,
                                            PortID logicalPort) const;
    bool sourceReady(const VpuMacroState &state, size_t sourceIndex,
                     PortID logicalPort) const;
    bool computeInputsReady(const VpuMacroState &state) const;
    bool storeSourceReady(const VpuMacroState &state) const;
    void writeResultBytes(const VpuMacroState &state, PortID dstPort,
                          const std::vector<uint8_t> &bytes) const;
    void appendStoreWhenReady(MacroCmdContext &macroCmd,
                              const VpuMacroState &state);
    void executeLegacy(VpuMacroState &state) const;
    void executeBinary(const VpuMacroState &state) const;
    void executeUnary(const VpuMacroState &state) const;
    void executeFma(const VpuMacroState &state) const;
    void executeReduce(const VpuMacroState &state) const;
    void executeLoadStoreBypass(const VpuMacroState &state) const;
    void executeVectorOp(VpuMacroState &state) const;

  protected:
    MacroCmdKind classifyMacroCmd(
        const std::vector<uint8_t> &cmd) const override;
    uint32_t classifyIssueQueue(const std::vector<uint8_t> &cmd,
                                MacroCmdKind kind) const override;
    void onMacroCmdBegin(MacroCmdContext &macroCmd) override;
    void buildUops(MacroCmdContext &macroCmd) override;
    void onMemUopComplete(MacroCmdContext &macroCmd,
                          const MemTxnContext &txn, PacketPtr pkt) override;
    void onExecUopComplete(MacroCmdContext &macroCmd,
                           const MicroOpContext &uop) override;
    void onMacroCmdEnd(MacroCmdContext &macroCmd) override;

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
