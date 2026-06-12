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

#ifndef __NPU_VPU_UNIT_HH__
#define __NPU_VPU_UNIT_HH__

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <unordered_map>
#include <vector>

#include "npu/LutUnit.hh"
#include "npu/SpecializedExecutionUnit.hh"
#include "params/VpuUnit.hh"

namespace gem5
{

class VpuUnit : public SpecializedExecutionUnit
{
  private:
    static constexpr uint8_t VpuDeviceType = 0x2;
    static constexpr Addr SpmBase = 0x60000000;
    static constexpr Addr SpmSlotStride = 0x40;

    static constexpr size_t FormatWord = 1;
    static constexpr size_t DstAddrWord = 2;
    static constexpr size_t DstShapeWord = 3;
    static constexpr size_t DstStrideWord = 4;
    static constexpr size_t Src0AddrWord = 5;
    static constexpr size_t Src0ShapeWord = 6;
    static constexpr size_t Src0StrideWord = 7;
    static constexpr size_t Src1AddrWord = 8;
    static constexpr size_t Src1ShapeWord = 9;
    static constexpr size_t Src1StrideWord = 10;
    static constexpr size_t Extra0Word = 11;

    enum class Opcode : uint8_t
    {
        VAdd = 0x1,
        VSub = 0x2,
        VMul = 0x3,
        VDiv = 0x4,
        VScale = 0x5,
        VCvtI2F = 0x6,
        VCvtF2I = 0x7,
        VSqrt = 0x8,
        VReduceSum = 0xa,
        VReduceMax = 0xb,
        VLoad = 0xc,
        VStore = 0xd,
        VExp = 0xe,
    };

    enum class DataType : uint32_t
    {
        Int8 = 0x0,
        Int16 = 0x1,
        Int32 = 0x2,
        UInt8 = 0x4,
        UInt16 = 0x5,
        UInt32 = 0x6,
        Float16 = 0x9,
        Float32 = 0xa,
    };

    enum class LayoutOrder : uint8_t
    {
        WC = 0,
        CW = 1,
    };

    enum class BufferRole : uint8_t
    {
        Input,
        Output,
    };

    struct AxisPair
    {
        uint32_t w = 1;
        uint32_t c = 1;
    };

    struct TensorDesc
    {
        Addr addr = 0;
        AxisPair shape;
        AxisPair stride;
    };

    struct DecodedVectorOp
    {
        Opcode opcode = Opcode::VAdd;
        DataType dstType = DataType::Int32;
        DataType src0Type = DataType::Int32;
        DataType src1Type = DataType::Int32;
        LayoutOrder layoutOrder = LayoutOrder::WC;
        uint32_t wLayoutElems = 1;
        uint32_t cLayoutElems = 1;
        uint32_t scalarBits = 0;
        size_t dstElemSize = sizeof(uint32_t);
        size_t src0ElemSize = sizeof(uint32_t);
        size_t src1ElemSize = sizeof(uint32_t);
        bool hasSrc1 = false;
        bool isReduce = false;
    };

    struct LocalAddr
    {
        BufferRole role = BufferRole::Input;
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
        TensorDesc dst;
        TensorDesc src0;
        TensorDesc src1;
        size_t dstSpanBytes = 0;
        size_t src0SpanBytes = 0;
        size_t src1SpanBytes = 0;
        bool src0InSpm = false;
        bool src1InSpm = false;
        bool dstInSpm = false;
        bool src0Loaded = false;
        bool src1Loaded = false;
        bool resultReady = false;
        bool storeIssued = false;
        uint32_t src0LoadChunks = 0;
        uint32_t src1LoadChunks = 0;
        uint32_t execChunks = 0;
        uint32_t storeChunks = 0;
        uint32_t nextSrc0LoadChunk = 0;
        uint32_t nextSrc1LoadChunk = 0;
        uint32_t nextExecChunk = 0;
        uint32_t nextStoreChunk = 0;
        uint32_t completedSrc0LoadChunks = 0;
        uint32_t completedSrc1LoadChunks = 0;
        uint32_t completedExecChunks = 0;
        uint32_t completedStoreChunks = 0;
        std::vector<uint8_t> src0Bytes;
        std::vector<uint8_t> src1Bytes;
        std::vector<uint8_t> resultBytes;
        uint32_t completedExecUops = 0;
    };

    enum class ExecToken : uint64_t
    {
        WaitStoreData = 1,
        RunCompute = 2,
        LoadSrc0 = 3,
        LoadSrc1 = 4,
        StoreResult = 5,
    };

    static constexpr uint32_t ChunkPipelineWindow = 64;

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
    const uint32_t dlenBytes;
    const uint32_t dlenGroupSize;
    const Cycles int8CyclesPerDlen;
    const Cycles int16CyclesPerDlen;
    const Cycles int32CyclesPerDlen;
    const Cycles float16CyclesPerDlen;
    const Cycles float32CyclesPerDlen;

    std::vector<LocalBufferSlot> inputBuffers;
    std::vector<LocalBufferSlot> outputBuffers;
    std::unordered_map<uint64_t, VpuMacroState> macroStates;

    Tick lastLinearExecuteLatencyValue = 0;
    Tick lastLinearCompletionTickValue = 0;

    uint32_t cmdWord(const std::vector<uint8_t> &cmd, size_t wordIdx) const;
    Addr slotAddr(PortID portId, Addr offset = 0) const;
    Opcode decodeOpcode(uint8_t opCode) const;
    DataType decodeDataType(uint32_t raw) const;
    AxisPair decodeAxisField(uint32_t raw) const;
    TensorDesc decodeTensor(const std::vector<uint8_t> &cmd,
                            size_t addrWord) const;
    size_t elemSizeBytes(DataType dataType) const;
    bool isFloatType(DataType dataType) const;
    bool isSignedType(DataType dataType) const;
    const char *opcodeName(Opcode opcode) const;
    const char *dataTypeName(DataType dataType) const;
    DecodedVectorOp decodeVectorOp(const MacroCmdContext &macroCmd) const;
    bool isComputeOpcode(Opcode opcode) const;
    bool isLutOpcode(Opcode opcode) const;
    bool isLinearOpcode(Opcode opcode) const;
    bool isSpmAddr(Addr addr) const;
    bool isInputLocalAddr(Addr addr) const;
    bool isOutputLocalAddr(Addr addr) const;
    LocalAddr decodeLocalAddr(Addr addr, BufferRole expected,
                              size_t accessSize) const;
    PortID decodeSpmPort(Addr addr, size_t accessSize) const;
    uint32_t layoutLowestDim(const DecodedVectorOp &op) const;
    size_t tileElems(const DecodedVectorOp &op) const;
    Cycles dtypeCyclesPerDlen(DataType dataType) const;
    uint32_t workDlenChunks(const VpuMacroState &state) const;
    uint32_t workDlenChunksInGroup(const VpuMacroState &state,
                                   uint32_t group) const;
    uint32_t byteDlenChunks(size_t bytes) const;
    size_t dlenGroupBytes() const;
    uint32_t byteDlenGroups(size_t bytes) const;
    size_t dlenChunkOffset(uint32_t chunk) const;
    size_t dlenChunkSize(size_t totalBytes, uint32_t chunk) const;
    size_t dlenGroupOffset(uint32_t group) const;
    size_t dlenGroupSizeBytes(size_t totalBytes, uint32_t group) const;
    size_t dlenGroupTimingBytes(size_t totalBytes, uint32_t group) const;
    PortID dlenChunkPort(Addr addr, size_t size, uint32_t chunk) const;
    PortID src0ReadPortId(uint32_t group) const;
    PortID src1ReadPortId(uint32_t group) const;
    PortID dstWritePortId(uint32_t group) const;
    uint64_t encodeToken(ExecToken token, uint32_t chunk = 0) const;
    ExecToken decodeTokenKind(uint64_t token) const;
    uint32_t decodeTokenChunk(uint64_t token) const;
    size_t tensorSpanBytes(const TensorDesc &tensor, size_t elemSize,
                           const DecodedVectorOp &op) const;
    size_t tensorElemOffset(const TensorDesc &tensor, const DecodedVectorOp &op,
                            uint32_t w, uint32_t c, size_t elemSize) const;
    AxisPair broadcastShape(const TensorDesc &src0, const TensorDesc &src1) const;
    void validateTensor(const TensorDesc &tensor, size_t elemSize,
                        const DecodedVectorOp &op, const char *label) const;
    void validateCommand(const MacroCmdContext &macroCmd,
                         VpuMacroState &state) const;
    Tick computeExecLatency(const VpuMacroState &state, uint32_t chunk);
    LutUnit::Operation lutOperation(Opcode opcode) const;
    LocalBufferSlot &bufferSlot(const LocalAddr &addr);
    const LocalBufferSlot &bufferSlot(const LocalAddr &addr) const;
    const std::vector<uint8_t> &sourceBytes(const TensorDesc &tensor,
                                            const DecodedVectorOp &op,
                                            size_t spanBytes) const;
    const std::vector<uint8_t> &macroSourceBytes(const VpuMacroState &state,
                                                 bool secondSource) const;
    bool sourceReady(const TensorDesc &tensor, BufferRole role,
                     size_t spanBytes) const;
    bool storeSourceReady(const VpuMacroState &state) const;
    bool computeSourcesReady(const VpuMacroState &state) const;
    void completeCompute(VpuMacroState &state);
    void writeResultBytes(const TensorDesc &dst,
                          const std::vector<uint8_t> &bytes,
                          size_t spanBytes) const;
    uint64_t loadUnsignedValue(const std::vector<uint8_t> &bytes, size_t offset,
                               size_t elemSize) const;
    int64_t loadSignedValue(const std::vector<uint8_t> &bytes, size_t offset,
                           size_t elemSize) const;
    double loadFloatValue(const std::vector<uint8_t> &bytes, size_t offset,
                          DataType dataType) const;
    void storeUnsignedValue(std::vector<uint8_t> &bytes, size_t offset,
                            uint64_t value, size_t elemSize) const;
    void storeSignedValue(std::vector<uint8_t> &bytes, size_t offset,
                          int64_t value, size_t elemSize) const;
    void storeFloatValue(std::vector<uint8_t> &bytes, size_t offset,
                         double value, DataType dataType) const;
    void executeBinary(VpuMacroState &state) const;
    void executeUnary(VpuMacroState &state) const;
    void executeReduce(VpuMacroState &state) const;
    void executeLoadStoreBypass(const VpuMacroState &state) const;
    void executeVectorOp(VpuMacroState &state) const;

  protected:
    MacroCmdKind classifyMacroCmd(
        const std::vector<uint8_t> &cmd) const override;
    uint32_t classifyIssueQueue(const std::vector<uint8_t> &cmd,
                                MacroCmdKind kind) const override;
    bool canActivateMacroCmd(const MacroCmdContext &macroCmd) const override;
    bool canIssueExecWithOutstandingMemUops(
        const MacroCmdContext &macroCmd,
        const MicroOpContext &uop) const override;
    void onMacroCmdBegin(MacroCmdContext &macroCmd) override;
    void buildUops(MacroCmdContext &macroCmd) override;
    void onMemUopComplete(MacroCmdContext &macroCmd,
                          const MemTxnContext &txn, PacketPtr pkt) override;
    void onExecUopComplete(MacroCmdContext &macroCmd,
                           const MicroOpContext &uop) override;
    void onMacroCmdEnd(MacroCmdContext &macroCmd) override;
    const char *profileSeuType() const override;
    void appendProfileDetailsJson(const MacroCmdContext &macroCmd,
                                  std::ostream &os) const override;

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

#endif // __NPU_VPU_UNIT_HH__
