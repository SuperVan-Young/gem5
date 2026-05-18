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
expectIntEq(const char *label, int32_t actual, int32_t expected)
{
    if (actual != expected) {
        printf("%s_FAIL actual=%d expected=%d\n", label, actual, expected);
        return false;
    }
    return true;
}

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
    static constexpr uint32_t Rows = 3U;
    static constexpr uint32_t Cols = 4U;
    static constexpr float Tolerance = 1.0e-5f;

    const std::array<float, Rows * Cols> src = {
        -1.0f, -0.5f, 0.5f, 1.0f,
        -200.0f, -100.0f, 63.0f, 200.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    std::array<int8_t, Rows * Cols> dst = {};
    std::array<PrimitiveScaleMetadata, Rows> metadata = {};

    primitiveQuantizeRowwiseF32ToI8(src.data(), Rows, Cols, dst.data(),
                                    metadata.data());

    const std::array<int32_t, Rows * Cols> expected = {
        -127, -64, 64, 127,
        -127, -64, 40, 127,
        0, 0, 0, 0,
    };

    for (uint32_t i = 0U; i < expected.size(); ++i) {
        if (!expectIntEq("FLASH_ATTENTION_QUANT_VALUE",
                         static_cast<int32_t>(dst[i]), expected[i])) {
            return 1;
        }
    }

    if (!expectFloatClose("FLASH_ATTENTION_QUANT_ROW0_SCALE",
                          metadata[0].inputToIntScale, 127.0f, Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_QUANT_ROW0_INV_SCALE",
                          metadata[0].intToInputScale, 1.0f / 127.0f,
                          Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_QUANT_ROW0_ABSMAX",
                          metadata[0].absMax, 1.0f, Tolerance)) {
        return 1;
    }

    if (!expectFloatClose("FLASH_ATTENTION_QUANT_ROW1_SCALE",
                          metadata[1].inputToIntScale, 127.0f / 200.0f,
                          Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_QUANT_ROW1_INV_SCALE",
                          metadata[1].intToInputScale, 200.0f / 127.0f,
                          Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_QUANT_ROW1_ABSMAX",
                          metadata[1].absMax, 200.0f, Tolerance)) {
        return 1;
    }

    if (!expectFloatClose("FLASH_ATTENTION_QUANT_ROW2_SCALE",
                          metadata[2].inputToIntScale, 0.0f, Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_QUANT_ROW2_INV_SCALE",
                          metadata[2].intToInputScale, 0.0f, Tolerance) ||
        !expectFloatClose("FLASH_ATTENTION_QUANT_ROW2_ABSMAX",
                          metadata[2].absMax, 0.0f, Tolerance)) {
        return 1;
    }

    printf("FLASH_ATTENTION_QUANTIZE_ROWWISE_F32_TO_I8_PASS\n");
    return 0;
}
