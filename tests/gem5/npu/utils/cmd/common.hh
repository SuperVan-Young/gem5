#ifndef TESTS_GEM5_NPU_UTILS_CMD_COMMON_H_
#define TESTS_GEM5_NPU_UTILS_CMD_COMMON_H_

#include <stdint.h>

#include <cstring>

#include "../npu_mmio.hh"

#define NPU_CMD_PORT_BASE 0x70000000UL
#define NPU_CMD_PORT_STRIDE (1UL << 20)
#define NPU_CMD_PORT_BASE_FOR(cpu_id) \
    (NPU_CMD_PORT_BASE + ((uint64_t)(cpu_id) * NPU_CMD_PORT_STRIDE))
#define NPU_CMD_BUFFER_BITS 512U
#define NPU_CMD_BUFFER_WORDS (NPU_CMD_BUFFER_BITS / 32U)
#define NPU_CMD_BUFFER_BYTES (NPU_CMD_BUFFER_WORDS * sizeof(uint32_t))
#define NPU_CMD_LAUNCH_WORDS NPU_CMD_BUFFER_WORDS
#define NPU_CMD_LAUNCH_BYTES (NPU_CMD_LAUNCH_WORDS * sizeof(uint32_t))
#define NPU_CMD_CTRL_ADDR(port_base) ((port_base) + NPU_CMD_LAUNCH_BYTES)
#define NPU_CMD_FAST_LAUNCH_ADDR(port_base) NPU_CMD_CTRL_ADDR(port_base)
#define NPU_CMD_CTRL_PUSH 0U
#define NPU_CMD_CTRL_POP 1U
#ifndef NPU_CMD_INLINE_BLOCK_LAUNCH
#define NPU_CMD_INLINE_BLOCK_LAUNCH 0
#endif

#define NPU_CMD_STAGE_CSR0 0x800U
#define NPU_CMD_STAGE_CSR1 0x801U
#define NPU_CMD_STAGE_CSR2 0x802U
#define NPU_CMD_STAGE_CSR3 0x803U
#define NPU_CMD_STAGE_CSR4 0x804U
#define NPU_CMD_STAGE_CSR5 0x805U
#define NPU_CMD_STAGE_CSR6 0x806U
#define NPU_CMD_STAGE_CSR7 0x807U

enum NpuDeviceType {
    NPU_DEVICE_TYPE_MEGA_CMD_QUEUE = 0x0U,
    NPU_DEVICE_TYPE_SYNC_INDICATOR_TABLE = 0x1U,
    NPU_DEVICE_TYPE_VPU = 0x2U,
    NPU_DEVICE_TYPE_MPU = 0x3U,
    NPU_DEVICE_TYPE_DMA = 0x4U,
};

union NpuCmdHeaderWord
{
    struct Bits
    {
        uint32_t reserved0 : 6;
        uint32_t setIndicatorSnd : 1;
        uint32_t setIndicatorSns : 1;
        uint32_t syncIndicator : 8;
        uint32_t opCode : 8;
        uint32_t deviceId : 4;
        uint32_t deviceType : 4;
    } bits;
    uint32_t raw;

    constexpr NpuCmdHeaderWord() : raw(0U) {}
};

struct NpuCmdBinaryData
{
    uint64_t rs[8];
};

static_assert(sizeof(NpuCmdHeaderWord) == sizeof(uint32_t),
              "NPU command header must remain 32 bits.");
static_assert(sizeof(NpuCmdBinaryData) == NPU_CMD_BUFFER_BYTES,
              "NPU command binary view must remain 64 bytes.");

static inline uint32_t
npuBuildHeaderWord(uint32_t device_type, uint32_t device_id, uint32_t op_code,
                   uint32_t sync_indicator, uint32_t set_indicator_sns,
                   uint32_t set_indicator_snd)
{
    NpuCmdHeaderWord header;

    header.bits.deviceType = device_type;
    header.bits.deviceId = device_id;
    header.bits.opCode = op_code;
    header.bits.syncIndicator = sync_indicator;
    header.bits.setIndicatorSns = set_indicator_sns;
    header.bits.setIndicatorSnd = set_indicator_snd;
    return header.raw;
}

__attribute__((naked, noinline, noclone, used))
static void
npuCmdLaunchRawPairsInsn(uint64_t p0, uint64_t p1, uint64_t p2, uint64_t p3,
                         uint64_t p4, uint64_t p5, uint64_t p6,
                         uint64_t p7)
{
    (void)p0;
    (void)p1;
    (void)p2;
    (void)p3;
    (void)p4;
    (void)p5;
    (void)p6;
    (void)p7;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        ".insn r 0x0b, 0, 1, x0, a0, a1\n\t"
        ".insn r 0x0b, 0, 2, x0, a2, a3\n\t"
        ".insn r 0x0b, 0, 3, x0, a4, a5\n\t"
        ".insn r 0x0b, 0, 4, x0, a6, a7\n\t"
        ".insn r 0x0b, 0, 0, x0, x0, x0\n\t"
        "addi x0, x0, 0\n\t"
        "jalr x0, 0(x1)\n\t"
        ".option pop\n\t");
}

__attribute__((naked, noinline, noclone, used))
static void
npuCmdLaunchInlineBlockInsn(const uint64_t *pairs)
{
    (void)pairs;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        "ld t0, 0(a0)\n\t"
        "ld t1, 8(a0)\n\t"
        "ld t2, 16(a0)\n\t"
        "ld t3, 24(a0)\n\t"
        "ld t4, 32(a0)\n\t"
        "ld t5, 40(a0)\n\t"
        "ld t6, 48(a0)\n\t"
        "ld a1, 56(a0)\n\t"
        ".insn r 0x0b, 0, 1, x0, t0, t1\n\t"
        ".insn r 0x0b, 0, 2, x0, t2, t3\n\t"
        ".insn r 0x0b, 0, 3, x0, t4, t5\n\t"
        ".insn r 0x0b, 0, 4, x0, t6, a1\n\t"
        ".insn r 0x0b, 0, 0, x0, x0, x0\n\t"
        "addi x0, x0, 0\n\t"
        "jalr x0, 0(x1)\n\t"
        ".option pop\n\t");
}

__attribute__((naked, noinline, noclone, used))
static void
npuCmdLaunchInsn()
{
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        ".insn r 0x0b, 0, 0, x0, x0, x0\n\t"
        "jalr x0, 0(x1)\n\t"
        ".option pop\n\t");
}

__attribute__((naked, noinline, noclone, used))
static void
npuCmdStageInsn0(uint64_t low, uint64_t high)
{
    (void)low;
    (void)high;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        ".insn r 0x0b, 0, 1, x0, a0, a1\n\t"
        "jalr x0, 0(x1)\n\t"
        ".option pop\n\t");
}

__attribute__((naked, noinline, noclone, used))
static void
npuCmdStageInsn1(uint64_t low, uint64_t high)
{
    (void)low;
    (void)high;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        ".insn r 0x0b, 0, 2, x0, a0, a1\n\t"
        "jalr x0, 0(x1)\n\t"
        ".option pop\n\t");
}

__attribute__((naked, noinline, noclone, used))
static void
npuCmdStageInsn2(uint64_t low, uint64_t high)
{
    (void)low;
    (void)high;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        ".insn r 0x0b, 0, 3, x0, a0, a1\n\t"
        "jalr x0, 0(x1)\n\t"
        ".option pop\n\t");
}

__attribute__((naked, noinline, noclone, used))
static void
npuCmdStageInsn3(uint64_t low, uint64_t high)
{
    (void)low;
    (void)high;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        ".insn r 0x0b, 0, 4, x0, a0, a1\n\t"
        "jalr x0, 0(x1)\n\t"
        ".option pop\n\t");
}

template <typename T>
static inline void
npuBinaryDataFromObject(NpuCmdBinaryData *binary, const T &value)
{
    static_assert(sizeof(T) == sizeof(NpuCmdBinaryData),
                  "Object size must match NPU command size.");

    *binary = {};
    memcpy(binary, &value, sizeof(*binary));
}

template <typename T>
static inline void
npuObjectFromBinaryData(T *value, const NpuCmdBinaryData &binary)
{
    static_assert(sizeof(T) == sizeof(NpuCmdBinaryData),
                  "Object size must match NPU command size.");

    *value = {};
    memcpy(value, &binary, sizeof(*value));
}

class NpuCmd
{
  public:
    NpuCmd() : storage{} {}

    void clear()
    {
        for (unsigned i = 0; i < 8U; ++i) {
            storage.pairs[i] = 0U;
        }
    }

    uint32_t getWord(unsigned word_index) const
    {
        if (word_index >= NPU_CMD_BUFFER_WORDS) {
            return 0U;
        }

        return storage.words[word_index];
    }

    void setWord(unsigned word_index, uint32_t value)
    {
        if (word_index < NPU_CMD_BUFFER_WORDS) {
            storage.words[word_index] = value;
        }
    }

    uint32_t getBits(unsigned lsb, unsigned width) const
    {
        if (width == 0U || width > 32U || lsb >= NPU_CMD_BUFFER_BITS ||
            width > (NPU_CMD_BUFFER_BITS - lsb)) {
            return 0U;
        }

        uint32_t value = 0U;
        unsigned bits_done = 0U;

        while (bits_done < width) {
            const unsigned bit_index = lsb + bits_done;
            const unsigned word_index = cmdWordIndex(bit_index);
            const unsigned bit_in_word = cmdBitIndex(bit_index);
            const unsigned chunk_width = minUnsigned(32U - bit_in_word,
                                                     width - bits_done);
            const uint32_t chunk =
                (storage.words[word_index] >> bit_in_word) &
                bitMask(chunk_width);

            value |= chunk << bits_done;
            bits_done += chunk_width;
        }

        return value;
    }

    void setBits(unsigned lsb, unsigned width, uint32_t value)
    {
        if (width == 0U || width > 32U || lsb >= NPU_CMD_BUFFER_BITS ||
            width > (NPU_CMD_BUFFER_BITS - lsb)) {
            return;
        }

        unsigned bits_done = 0U;

        while (bits_done < width) {
            const unsigned bit_index = lsb + bits_done;
            const unsigned word_index = cmdWordIndex(bit_index);
            const unsigned bit_in_word = cmdBitIndex(bit_index);
            const unsigned chunk_width = minUnsigned(32U - bit_in_word,
                                                     width - bits_done);
            const uint32_t field_mask = bitMask(chunk_width) << bit_in_word;
            const uint32_t chunk = ((value >> bits_done) &
                                    bitMask(chunk_width)) << bit_in_word;

            storage.words[word_index] =
                (storage.words[word_index] & ~field_mask) | chunk;
            bits_done += chunk_width;
        }
    }

    uint32_t getDeviceType() const { return getBits(28U, 4U); }
    void setDeviceType(uint32_t value) { setBits(28U, 4U, value); }

    uint32_t getDeviceId() const { return getBits(24U, 4U); }
    void setDeviceId(uint32_t value) { setBits(24U, 4U, value); }

    uint32_t getOpCode() const { return getBits(16U, 8U); }
    void setOpCode(uint32_t value) { setBits(16U, 8U, value); }

    uint32_t getSyncIndicator() const { return getBits(8U, 8U); }
    void setSyncIndicator(uint32_t value) { setBits(8U, 8U, value); }

    uint32_t getSetIndicatorSns() const { return getBits(7U, 1U); }
    void setSetIndicatorSns(uint32_t value) { setBits(7U, 1U, value); }

    uint32_t getSetIndicatorSnd() const { return getBits(6U, 1U); }
    void setSetIndicatorSnd(uint32_t value) { setBits(6U, 1U, value); }

    void clearCommonReservedBits() { setBits(0U, 6U, 0U); }

    void loadBinary(const NpuCmdBinaryData &binary)
    {
        for (unsigned i = 0; i < 8U; ++i) {
            storage.pairs[i] = binary.rs[i];
        }
    }

    void copyBinaryData(NpuCmdBinaryData *binary) const
    {
        *binary = {};
        for (unsigned i = 0; i < 8U; ++i) {
            binary->rs[i] = storage.pairs[i];
        }
    }

    void writeCmdWordsAt(unsigned word_count, uint64_t port_base) const
    {
        npu_mmio_write32(port_base, storage.words, word_count);
    }

    void ringDoorbellAt(uint64_t port_base) const
    {
        npu_mmio_write32_one(NPU_CMD_CTRL_ADDR(port_base), NPU_CMD_CTRL_PUSH);
    }

    void launchCmdWordsAt(unsigned word_count, uint64_t port_base) const
    {
        writeCmdWordsAt(word_count, port_base);
        ringDoorbellAt(port_base);
    }

    void launchCmdViaMmioAt(uint64_t port_base) const
    {
        launchCmdWordsAt(NPU_CMD_LAUNCH_WORDS, port_base);
    }

    void launchCmdViaMmio() const
    {
        launchCmdViaMmioAt(NPU_CMD_PORT_BASE);
    }

    void launchCmdAt(uint64_t port_base) const
    {
        launchCmdViaStage2At(port_base);
    }

    void launchCmd() const
    {
        launchCmdAt(NPU_CMD_PORT_BASE);
    }

    void stageCmdWords() const
    {
        stageBinaryPairs(storage.pairs);
    }

    void launchStagedCmdAt(uint64_t port_base) const
    {
        (void)port_base;
        asm volatile(".insn r 0x0b, 0, 0, x0, x0, x0" : : : "memory");
    }

    void launchCmdViaStage2At(uint64_t port_base) const
    {
#if NPU_CMD_INLINE_BLOCK_LAUNCH
        launchBinaryPairsInlineBlockAt(storage.pairs, port_base);
#else
        launchBinaryPairsUnrolledAt(storage.pairs, port_base);
#endif
    }

    void launchCmdViaStage2() const
    {
        launchCmdViaStage2At(NPU_CMD_PORT_BASE);
    }

  private:
    static void stageBinaryPairs(const uint64_t *pairs)
    {
        for (unsigned i = 0; i < 4U; ++i) {
            writeStagePair(i, pairs[(2U * i)], pairs[(2U * i) + 1U]);
        }
    }

    static void launchBinaryPairsAt(const uint64_t *pairs, uint64_t port_base)
    {
        stageBinaryPairs(pairs);
        launchStagedOnlyAt(port_base);
    }

    static void stageBinaryPairsUnrolled(const uint64_t *pairs)
    {
        npuCmdStageInsn0(pairs[0], pairs[1]);
        npuCmdStageInsn1(pairs[2], pairs[3]);
        npuCmdStageInsn2(pairs[4], pairs[5]);
        npuCmdStageInsn3(pairs[6], pairs[7]);
    }

    static void
    launchBinaryPairsUnrolledAt(const uint64_t *pairs, uint64_t port_base)
    {
        stageBinaryPairsUnrolled(pairs);
        asm volatile("" ::: "memory");
        launchStagedOnlyAt(port_base);
    }

    static void
    launchBinaryPairsInlineBlockAt(const uint64_t *pairs, uint64_t port_base)
    {
        (void)port_base;
        asm volatile("" ::: "memory");
        npuCmdLaunchInlineBlockInsn(pairs);
        asm volatile("" ::: "memory");
    }

    static void launchStagedOnlyAt(uint64_t port_base)
    {
        (void)port_base;
        asm volatile("" ::: "memory");
        npuCmdLaunchInsn();
        asm volatile("" ::: "memory");
    }

    static void
    writeStagePair(unsigned pair_index, uint64_t low, uint64_t high)
    {
        switch (pair_index) {
          case 0:
            asm volatile("" ::: "memory");
            npuCmdStageInsn0(low, high);
            asm volatile("" ::: "memory");
            break;
          case 1:
            asm volatile("" ::: "memory");
            npuCmdStageInsn1(low, high);
            asm volatile("" ::: "memory");
            break;
          case 2:
            asm volatile("" ::: "memory");
            npuCmdStageInsn2(low, high);
            asm volatile("" ::: "memory");
            break;
          case 3:
            asm volatile("" ::: "memory");
            npuCmdStageInsn3(low, high);
            asm volatile("" ::: "memory");
            break;
          default:
            break;
        }
    }

    static unsigned cmdWordIndex(unsigned bit_index)
    {
        return bit_index / 32U;
    }

    static unsigned cmdBitIndex(unsigned bit_index)
    {
        return bit_index & 31U;
    }

    static unsigned minUnsigned(unsigned lhs, unsigned rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    static uint32_t bitMask(unsigned width)
    {
        if (width >= 32U) {
            return 0xFFFFFFFFU;
        }

        return (1U << width) - 1U;
    }

    union Storage
    {
        uint32_t words[NPU_CMD_BUFFER_WORDS];
        uint64_t pairs[8];

        constexpr Storage() : pairs{} {}
    } storage;
};

#endif
