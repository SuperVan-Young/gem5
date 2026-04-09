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

#ifndef __NPU_LUT_UNIT_HH__
#define __NPU_LUT_UNIT_HH__

#include <cstdint>
#include <vector>

#include "base/types.hh"
#include "params/LutUnit.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class LutUnit : public SimObject
{
  public:
    enum class Operation : uint8_t
    {
        Sqrt,
        Exp,
        Softmax,
    };

  private:
    struct ResourceState
    {
        Tick availableTick = 0;
        uint64_t requestCount = 0;
        uint64_t commandCount = 0;
    };

    const Tick rangeReductionLatency;
    const Tick lookupLatency;
    const Tick interpolationLatency;
    const Tick normalizeLatency;
    const uint32_t tableEntries;
    std::vector<float> sqrtTable;
    std::vector<float> expTable;
    ResourceState resource;
    Tick lastExecuteLatencyValue = 0;
    Tick lastCompletionTickValue = 0;
    Tick lastSoftmaxExecuteLatencyValue = 0;
    Tick lastSoftmaxCompletionTickValue = 0;

    void buildLookupTables();
    float interpolateTableValue(const std::vector<float> &table,
                                double normalizedPosition) const;

  public:
    LutUnit(const LutUnitParams &params);

    Tick reserve(Operation op, uint32_t requestCount, Tick now);
    void noteCompletion(Operation op, Tick tick);
    float evaluateSqrt(float value) const;
    float evaluateExp(float value) const;

    uint64_t requestCount() const;
    uint64_t commandCount() const;
    Tick lastExecuteLatency() const;
    Tick lastCompletionTick() const;
    Tick lastSoftmaxExecuteLatency() const;
    Tick lastSoftmaxCompletionTick() const;
};

} // namespace gem5

#endif // __NPU_LUT_UNIT_HH__
