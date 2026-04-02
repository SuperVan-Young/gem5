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

#ifndef __NPU_MEGA_MPU_UNIT_HH__
#define __NPU_MEGA_MPU_UNIT_HH__

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "npu/mega/SpecializedExecutionUnit.hh"
#include "params/MpuUnit.hh"

namespace gem5
{

class MpuUnit : public SpecializedExecutionUnit
{
  private:
    enum class Mode : uint8_t
    {
        Load = 0,
        Compute = 1,
        Store = 2,
        TensorLoop = 4,
    };

    enum class SlotKind : uint8_t
    {
        A = 0,
        B = 1,
        C = 2,
    };

    enum class ComputeSubop : uint32_t
    {
        Matmul = 0,
        MatmulAcc = 1,
    };

    enum class Axis : uint32_t
    {
        M = 0,
        N = 1,
        K = 2,
    };

    enum class PendingLoadKind : uint8_t
    {
        SlotLoad,
        TensorA,
        TensorB,
    };

    enum SlotIndex : int
    {
        SlotA0 = 0,
        SlotA1 = 1,
        SlotB0 = 2,
        SlotB1 = 3,
        SlotC0 = 4,
        SlotC1 = 5,
    };

    struct ParsedCmd
    {
        std::array<uint32_t, 16> words = {};
        uint8_t deviceId = 0;
        uint8_t dataType = 0;
        uint8_t mode = 0;
        uint8_t syncIndicator = 0;
        bool setIndicatorSns = false;
        bool setIndicatorSnd = false;

        uint32_t localAddrA = 0;
        uint32_t localAddrB = 0;
        uint32_t localAddrC = 0;
        Addr spmAddrA = 0;
        Addr spmAddrB = 0;
        Addr spmAddrC = 0;
        uint32_t validM = 0;
        uint32_t validN = 0;
        uint32_t validK = 0;
        uint32_t layoutMode = 0;
        ComputeSubop subop = ComputeSubop::Matmul;
        uint32_t computeModeFlags = 0;
        Axis outerAxis = Axis::M;
        Axis innerAxis = Axis::N;
        uint32_t outerCount = 0;
        uint32_t innerCount = 0;
        uint32_t outerStepTiles = 0;
        uint32_t innerStepTiles = 0;
        bool pingpongA = false;
        bool pingpongB = false;
        bool pingpongC = false;
        bool autoLoadCForAcc = false;
    };

    struct SlotState
    {
        SlotKind kind = SlotKind::A;
        uint32_t localAddr = 0;
        bool valid = false;
        bool busy = false;
        bool dirty = false;
        uint32_t shapeM = 0;
        uint32_t shapeN = 0;
        uint32_t shapeK = 0;
        std::vector<uint8_t> bytes;
    };

    struct IterationPlan
    {
        uint32_t validM = 0;
        uint32_t validN = 0;
        uint32_t validK = 0;
        int loadSlotIndex = -1;
        int computeSlotA = -1;
        int computeSlotB = -1;
        int computeSlotC = -1;
        int storeSlotIndex = -1;
        Addr loadSpmAddr = 0;
        Addr storeSpmAddr = 0;
        Addr tensorAAddr = 0;
        Addr tensorBAddr = 0;
        Addr tensorCAddr = 0;
        std::vector<uint8_t> tensorA;
        std::vector<uint8_t> tensorB;
        std::vector<uint8_t> tensorC;
        ComputeSubop subop = ComputeSubop::Matmul;
    };

    struct PendingLoadTxn
    {
        uint64_t iteration = 0;
        PendingLoadKind kind = PendingLoadKind::SlotLoad;
        int slotIndex = -1;
    };

    static constexpr uint8_t MpuDeviceType = 0x3;
    static constexpr Addr SpmBase = 0x60000000ULL;
    static constexpr Addr SpmEnd = 0x6fffffffULL;
    static constexpr uint32_t LayoutModeNormal = 0;
    static constexpr size_t Int8Bytes = 1;
    static constexpr size_t Int32Bytes = 4;

    const uint32_t arrayRows;
    const uint32_t arrayCols;
    const uint32_t arrayKDepth;
    const Tick loadBaseLatency;
    const Tick storeBaseLatency;
    const Tick arrayFillLatency;
    const Tick arraySteadyPerK;
    const Tick arrayDrainLatency;

    ParsedCmd parsedCmd;
    bool parsedCmdValid;
    std::array<SlotState, 6> slots;
    std::vector<IterationPlan> iterationPlans;
    std::unordered_map<uint64_t, PendingLoadTxn> pendingLoadTxns;

    uint64_t loadCount;
    uint64_t computeCount;
    uint64_t storeCount;
    uint64_t tensorLoopCount;
    uint64_t matmulCountValue;
    uint64_t matmulAccCountValue;
    uint64_t tensorLoopExpandedTilesCount;

    uint32_t extractWord(const std::vector<uint8_t> &cmd, size_t index) const;
    ParsedCmd parseCommand(const std::vector<uint8_t> &cmd) const;
    void resetCommandState();

    void validateParsedCommand(const ParsedCmd &cmd) const;
    void validateValidShape(const ParsedCmd &cmd) const;
    void validateLoadCommand(const ParsedCmd &cmd) const;
    void validateComputeCommand(const ParsedCmd &cmd) const;
    void validateStoreCommand(const ParsedCmd &cmd) const;
    void validateTensorLoopCommand(const ParsedCmd &cmd) const;

    bool spmContains(Addr addr, size_t size) const;
    void validateSpmAddress(Addr addr, size_t size, const char *label) const;

    int slotIndexForAddr(uint32_t local_addr) const;
    SlotState &slotByIndex(int slot_index);
    const SlotState &slotByIndex(int slot_index) const;
    void validateSlotAddress(uint32_t local_addr, SlotKind expected_kind,
                             const char *label) const;
    void validateSlotShape(const SlotState &slot, const ParsedCmd &cmd,
                           const char *label) const;
    size_t slotCapacityBytes(int slot_index) const;
    size_t tileBytesForSlotKind(SlotKind kind, uint32_t valid_m, uint32_t valid_n,
                                uint32_t valid_k) const;
    size_t aTileBytes(uint32_t valid_m, uint32_t valid_k) const;
    size_t bTileBytes(uint32_t valid_k, uint32_t valid_n) const;
    size_t cTileBytes(uint32_t valid_m, uint32_t valid_n) const;
    Tick matmulLatency(uint32_t valid_k) const;
    Tick matmulAccLatency(uint32_t valid_k) const;
    void setSlotShape(SlotState &slot, const ParsedCmd &cmd);
    void clearSlotBusy(int slot_index);
    void markBusyForFineCommand(const ParsedCmd &cmd);
    void clearBusyForFineCommand(const ParsedCmd &cmd);
    void writeInt32(std::vector<uint8_t> &bytes, size_t element_index,
                    int32_t value) const;
    int32_t readInt32(const std::vector<uint8_t> &bytes,
                      size_t element_index) const;
    void runMatmul(const std::vector<uint8_t> &a_bytes,
                   const std::vector<uint8_t> &b_bytes,
                   std::vector<uint8_t> &c_bytes, uint32_t valid_m,
                   uint32_t valid_n, uint32_t valid_k, bool accumulate) const;

    void buildIterationPlans(ActiveExecution &exec);
    void buildTensorLoopPlans();
    IterationPlan &iterationPlan(uint64_t iteration);
    const IterationPlan *findIterationPlan(uint64_t iteration) const;
    Addr axisDeltaA(Axis axis, uint32_t valid_m, uint32_t valid_n,
                    uint32_t valid_k) const;
    Addr axisDeltaB(Axis axis, uint32_t valid_m, uint32_t valid_n,
                    uint32_t valid_k) const;
    Addr axisDeltaC(Axis axis, uint32_t valid_m, uint32_t valid_n,
                    uint32_t valid_k) const;

  protected:
    void startExecuteCommand(const std::vector<uint8_t> &cmd) override;
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

  public:
    MpuUnit(const MpuUnitParams &params);

    uint64_t loadCmdCount() const;
    uint64_t computeCmdCount() const;
    uint64_t storeCmdCount() const;
    uint64_t tensorLoopCmdCount() const;
    uint64_t matmulCount() const;
    uint64_t matmulAccCount() const;
    uint64_t slotA0Valid() const;
    uint64_t slotA1Valid() const;
    uint64_t slotB0Valid() const;
    uint64_t slotB1Valid() const;
    uint64_t slotC0Valid() const;
    uint64_t slotC1Valid() const;
    uint64_t slotC0Dirty() const;
    uint64_t slotC1Dirty() const;
    uint64_t tensorLoopExpandedTiles() const;
};

} // namespace gem5

#endif // __NPU_MEGA_MPU_UNIT_HH__
