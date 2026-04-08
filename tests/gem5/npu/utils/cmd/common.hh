#ifndef TESTS_GEM5_NPU_UTILS_CMD_COMMON_H_
#define TESTS_GEM5_NPU_UTILS_CMD_COMMON_H_

#include <stdint.h>

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

class NpuCmd
{
  public:
    NpuCmd() : words{} {}

    void clear()
    {
        for (unsigned i = 0; i < NPU_CMD_BUFFER_WORDS; ++i) {
            words[i] = 0U;
        }
    }

    uint32_t getWord(unsigned word_index) const
    {
        if (word_index >= NPU_CMD_BUFFER_WORDS) {
            return 0U;
        }

        return words[word_index];
    }

    void setWord(unsigned word_index, uint32_t value)
    {
        if (word_index < NPU_CMD_BUFFER_WORDS) {
            words[word_index] = value;
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
            const uint32_t chunk = (words[word_index] >> bit_in_word) &
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

            words[word_index] = (words[word_index] & ~field_mask) | chunk;
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

    void writeCmdWordsAt(unsigned word_count, uint64_t port_base) const
    {
        npu_mmio_write32(port_base, words, word_count);
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

    void launchCmdAt(uint64_t port_base) const
    {
        launchCmdWordsAt(NPU_CMD_LAUNCH_WORDS, port_base);
    }

    void launchCmd() const
    {
        launchCmdAt(NPU_CMD_PORT_BASE);
    }

    void stageCmdWords() const
    {
        for (unsigned i = 0; i < (NPU_CMD_BUFFER_WORDS / 2U); ++i) {
            const uint64_t packed =
                ((uint64_t)words[(2U * i) + 1U] << 32) | words[2U * i];
            writeStageCsr(i, packed);
        }
    }

    void launchStagedCmdAt(uint64_t port_base) const
    {
        (void)port_base;
        asm volatile(".insn r 0x0b, 0, 0, x0, x0, x0" : : : "memory");
    }

    void launchCmdViaStage2At(uint64_t port_base) const
    {
        stageCmdWords();
        launchStagedCmdAt(port_base);
    }

    void launchCmdViaStage2() const
    {
        launchCmdViaStage2At(NPU_CMD_PORT_BASE);
    }

  private:
    static void writeStageCsr(unsigned stage_index, uint64_t value)
    {
        switch (stage_index) {
          case 0:
            asm volatile("csrw 0x800, %0" : : "r"(value) : "memory");
            break;
          case 1:
            asm volatile("csrw 0x801, %0" : : "r"(value) : "memory");
            break;
          case 2:
            asm volatile("csrw 0x802, %0" : : "r"(value) : "memory");
            break;
          case 3:
            asm volatile("csrw 0x803, %0" : : "r"(value) : "memory");
            break;
          case 4:
            asm volatile("csrw 0x804, %0" : : "r"(value) : "memory");
            break;
          case 5:
            asm volatile("csrw 0x805, %0" : : "r"(value) : "memory");
            break;
          case 6:
            asm volatile("csrw 0x806, %0" : : "r"(value) : "memory");
            break;
          case 7:
            asm volatile("csrw 0x807, %0" : : "r"(value) : "memory");
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

    uint32_t words[NPU_CMD_BUFFER_WORDS];
};

#endif
