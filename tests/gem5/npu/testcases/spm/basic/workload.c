/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#define SPM_BASE 0x60000000

volatile unsigned char *spm = (volatile unsigned char *)SPM_BASE;
volatile unsigned int *spm_words = (volatile unsigned int *)SPM_BASE;

static void
panic(void)
{
    volatile int *dummy = (volatile int *)0xDEADBEEF;
    *dummy = 0;
    while (1) {
    }
}

static int
test_byte_access(void)
{
    int pass_count = 0;

    for (int i = 0; i < 256; ++i) {
        spm[i] = (unsigned char)(i & 0xFF);
    }

    for (int i = 0; i < 256; ++i) {
        unsigned char expected = (unsigned char)(i & 0xFF);
        if (spm[i] != expected) {
            panic();
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

static int
test_word_access(void)
{
    int pass_count = 0;

    for (int i = 0; i < 64; ++i) {
        spm_words[i] = (unsigned int)(0xDEAD0000 | (i & 0xFFFF));
    }

    for (int i = 0; i < 64; ++i) {
        unsigned int expected = (unsigned int)(0xDEAD0000 | (i & 0xFFFF));
        if (spm_words[i] != expected) {
            panic();
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

static int
test_overwrite(void)
{
    int pass_count = 0;

    for (int i = 0; i < 128; ++i) {
        spm[i] = 0xAA;
    }

    for (int i = 0; i < 128; ++i) {
        spm[i] = 0x55;
    }

    for (int i = 0; i < 128; ++i) {
        if (spm[i] != 0x55) {
            panic();
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

static int
test_unaligned_access(void)
{
    int pass_count = 0;

    for (int i = 0; i < 16; ++i) {
        spm[i + 1] = (unsigned char)(i * 2);
    }

    for (int i = 0; i < 16; ++i) {
        if (spm[i + 1] != (unsigned char)(i * 2)) {
            panic();
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

int
main(void)
{
    int byte_pass = test_byte_access();
    int word_pass = test_word_access();
    int overwrite_pass = test_overwrite();
    int unalign_pass = test_unaligned_access();

    return (
        byte_pass > 0 &&
        word_pass > 0 &&
        overwrite_pass > 0 &&
        unalign_pass > 0
    ) ? 0 : 1;
}
