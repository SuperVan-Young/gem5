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

#include "npu/LutUnit.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include "base/logging.hh"

namespace gem5
{

namespace
{

constexpr double Ln2 = 0.693147180559945309417232121458176568;

int
floorDiv2(int value)
{
    return value >= 0 ? value / 2 : -(((-value) + 1) / 2);
}

} // anonymous namespace

LutUnit::LutUnit(const LutUnitParams &params)
    : SimObject(params),
      rangeReductionLatency(params.range_reduction_latency),
      lookupLatency(params.lookup_latency),
      interpolationLatency(params.interpolation_latency),
      normalizeLatency(params.normalize_latency),
      dlenBytes(params.dlen_bytes),
      tableEntries(params.table_entries)
{
    panic_if(dlenBytes == 0, "%s: dlen_bytes must be non-zero", name());
    panic_if(tableEntries < 2,
             "%s: table_entries must be at least 2 (got %u)",
             name(), tableEntries);
    buildLookupTables();
}

void
LutUnit::buildLookupTables()
{
    sqrtTable.resize(tableEntries, 0.0f);
    expTable.resize(tableEntries, 0.0f);

    for (uint32_t idx = 0; idx < tableEntries; ++idx) {
        const double ratio =
            static_cast<double>(idx) / static_cast<double>(tableEntries - 1);
        sqrtTable[idx] = std::sqrt(1.0 + (3.0 * ratio));
        expTable[idx] = std::exp(Ln2 * ratio);
    }
}

float
LutUnit::interpolateTableValue(const std::vector<float> &table,
                               double normalizedPosition) const
{
    const double clamped = std::clamp(normalizedPosition, 0.0, 1.0);
    const double scaled = clamped * static_cast<double>(table.size() - 1);
    const auto lower = static_cast<size_t>(scaled);
    const size_t upper = std::min(lower + 1, table.size() - 1);
    const double fraction = scaled - static_cast<double>(lower);
    return static_cast<float>((1.0 - fraction) * table[lower] +
                              fraction * table[upper]);
}

Tick
LutUnit::reserve(Operation op, size_t workBytes, Tick now)
{
    if (workBytes == 0) {
        lastExecuteLatencyValue = 0;
        if (op == Operation::Softmax) {
            lastSoftmaxExecuteLatencyValue = 0;
        }
        return 0;
    }

    const Tick perRequestLatency = rangeReductionLatency + lookupLatency +
        interpolationLatency + normalizeLatency;
    const uint32_t requestCount = std::max<uint32_t>(
        1U, static_cast<uint32_t>((workBytes + dlenBytes - 1U) / dlenBytes));
    const Tick start = std::max(resource.availableTick, now);
    const Tick completion = start + (perRequestLatency * requestCount);
    resource.availableTick = completion;

    resource.requestCount += requestCount;
    resource.commandCount++;
    lastExecuteLatencyValue = completion - now;
    if (op == Operation::Softmax) {
        lastSoftmaxExecuteLatencyValue = lastExecuteLatencyValue;
    }

    return lastExecuteLatencyValue;
}

void
LutUnit::noteCompletion(Operation op, Tick tick)
{
    lastCompletionTickValue = tick;
    if (op == Operation::Softmax) {
        lastSoftmaxCompletionTickValue = tick;
    }
}

float
LutUnit::evaluateSqrt(float value) const
{
    if (std::isnan(value)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (value < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (value == 0.0f || std::isinf(value)) {
        return value;
    }

    const int exp2 = std::ilogb(value);
    const int scale = floorDiv2(exp2);
    const float reduced = std::scalbn(value, -2 * scale);
    const double position = (static_cast<double>(reduced) - 1.0) / 3.0;
    const float normalized = interpolateTableValue(sqrtTable, position);
    return std::scalbn(normalized, scale);
}

float
LutUnit::evaluateExp(float value) const
{
    if (std::isnan(value)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (value == std::numeric_limits<float>::infinity()) {
        return value;
    }
    if (value == -std::numeric_limits<float>::infinity()) {
        return 0.0f;
    }

    int exponent = static_cast<int>(std::floor(value / Ln2));
    double reduced = static_cast<double>(value) -
        (static_cast<double>(exponent) * Ln2);
    if (reduced < 0.0) {
        reduced += Ln2;
        exponent -= 1;
    }

    const double position = reduced / Ln2;
    const float normalized = interpolateTableValue(expTable, position);
    return std::ldexp(normalized, exponent);
}

uint32_t
LutUnit::dlenBytesValue() const
{
    return dlenBytes;
}

uint64_t
LutUnit::requestCount() const
{
    return resource.requestCount;
}

uint64_t
LutUnit::commandCount() const
{
    return resource.commandCount;
}

Tick
LutUnit::lastExecuteLatency() const
{
    return lastExecuteLatencyValue;
}

Tick
LutUnit::lastCompletionTick() const
{
    return lastCompletionTickValue;
}

Tick
LutUnit::lastSoftmaxExecuteLatency() const
{
    return lastSoftmaxExecuteLatencyValue;
}

Tick
LutUnit::lastSoftmaxCompletionTick() const
{
    return lastSoftmaxCompletionTickValue;
}

} // namespace gem5
