/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <array>

#include "quantize.hh"

namespace
{

static bool
expectFloatClose(const char *label, float actual, float expected,
                 float tolerance)
{
    if (fabsf(actual - expected) > tolerance) {
        printf("%s_FAIL actual=%.8f expected=%.8f\n", label, actual, expected);
        return false;
    }
    return true;
}

} // namespace

int
main(void)
{
    static constexpr float Tolerance = 1.0e-5f;
    static constexpr float Scale = 0.25f;
    const std::array<int32_t, 6> src = {-508, -127, 0, 128, 255, 508};
    std::array<float, src.size()> dst = {};

    const PrimitiveScaleMetadata metadata =
        primitiveDequantizeI32ToF32(src.data(), src.size(), Scale, dst.data());
    const std::array<float, src.size()> expected = {
        -127.0f, -31.75f, 0.0f, 32.0f, 63.75f, 127.0f,
    };

    for (uint32_t i = 0U; i < expected.size(); ++i) {
        if (!expectFloatClose("FLASH_ATTENTION_DEQUANT_VALUE", dst[i],
                              expected[i], Tolerance)) {
            return 1;
        }
    }

    if (!expectFloatClose("FLASH_ATTENTION_DEQUANT_SCALE",
                          metadata.intToInputScale, Scale, Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_DEQUANT_INV_SCALE",
                          metadata.inputToIntScale, 4.0f, Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_DEQUANT_ABSMAX", metadata.absMax,
                          127.0f, Tolerance)) {
        return 1;
    }

    const std::array<int32_t, 3> zero_src = {1, -5, 9};
    std::array<float, zero_src.size()> zero_dst = {};
    const PrimitiveScaleMetadata zero_meta = primitiveDequantizeI32ToF32(
        zero_src.data(), zero_src.size(), 0.0f, zero_dst.data());
    for (uint32_t i = 0U; i < zero_dst.size(); ++i) {
        if (!expectFloatClose("FLASH_ATTENTION_DEQUANT_ZERO_SCALE_VALUE",
                              zero_dst[i], 0.0f, Tolerance)) {
            return 1;
        }
    }
    if (!expectFloatClose("FLASH_ATTENTION_DEQUANT_ZERO_SCALE",
                          zero_meta.intToInputScale, 0.0f, Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_DEQUANT_ZERO_INV_SCALE",
                          zero_meta.inputToIntScale, 0.0f, Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_DEQUANT_ZERO_ABSMAX",
                          zero_meta.absMax, 0.0f, Tolerance)) {
        return 1;
    }

    printf("FLASH_ATTENTION_DEQUANTIZE_I32_TO_F32_PASS\n");
    return 0;
}
