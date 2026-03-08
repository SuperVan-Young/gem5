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

/*
 * Scratchpad Memory (SPM) test program for RISC-V
 *
 * This program tests the ScratchpadMemory by performing read and write
 * operations and verifying data consistency.
 */

#define SPM_BASE 0x60000000
#define SPM_SIZE (64 * 1024)  /* 64KB test region */

/* Volatile pointer for SPM access */
volatile unsigned char *spm = (volatile unsigned char *)SPM_BASE;
volatile unsigned int *spm_words = (volatile unsigned int *)SPM_BASE;

static void panic(const char *msg)
{
    /* In gem5 SE mode, we can't print easily, so use a loop to indicate failure */
    volatile int *dummy = (volatile int *)0xDEADBEEF;
    *dummy = 0;
    while (1);
}

static int test_byte_access(void)
{
    int i;
    unsigned char expected;
    int pass_count = 0;
    int test_size = 256;  /* Test first 256 bytes */

    /* Write pattern */
    for (i = 0; i < test_size; i++) {
        spm[i] = (unsigned char)(i & 0xFF);
    }

    /* Read and verify */
    for (i = 0; i < test_size; i++) {
        expected = (unsigned char)(i & 0xFF);
        if (spm[i] != expected) {
            panic("Byte access mismatch");
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

static int test_word_access(void)
{
    int i;
    unsigned int expected;
    int pass_count = 0;
    int test_size = 64;  /* Test 64 words (256 bytes) */

    /* Write pattern */
    for (i = 0; i < test_size; i++) {
        spm_words[i] = (unsigned int)(0xDEAD0000 | (i & 0xFFFF));
    }

    /* Read and verify */
    for (i = 0; i < test_size; i++) {
        expected = (unsigned int)(0xDEAD0000 | (i & 0xFFFF));
        if (spm_words[i] != expected) {
            panic("Word access mismatch");
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

static int test_overwrite(void)
{
    int i;
    int pass_count = 0;
    int test_size = 128;

    /* Write first pattern */
    for (i = 0; i < test_size; i++) {
        spm[i] = 0xAA;
    }

    /* Overwrite with second pattern */
    for (i = 0; i < test_size; i++) {
        spm[i] = 0x55;
    }

    /* Verify second pattern */
    for (i = 0; i < test_size; i++) {
        if (spm[i] != 0x55) {
            panic("Overwrite mismatch");
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

static int test_unaligned_access(void)
{
    int i;
    int pass_count = 0;

    /* Write at offset 1 (unaligned for word access) */
    for (i = 0; i < 16; i++) {
        spm[i + 1] = (unsigned char)(i * 2);
    }

    /* Verify */
    for (i = 0; i < 16; i++) {
        if (spm[i + 1] != (unsigned char)(i * 2)) {
            panic("Unaligned access mismatch");
            return 0;
        }
        pass_count++;
    }

    return pass_count;
}

int main(void)
{
    int byte_pass, word_pass, over_pass, unalign_pass;
    int total_pass;

    /* Run all tests */
    byte_pass = test_byte_access();
    word_pass = test_word_access();
    over_pass = test_overwrite();
    unalign_pass = test_unaligned_access();

    total_pass = byte_pass + word_pass + over_pass + unalign_pass;

    /* Print result marker for test verification */
    /* In SE mode, we use the return value to indicate pass/fail */

    /* Return the total number of passed sub-tests as indicator */
    return (byte_pass > 0 && word_pass > 0 && over_pass > 0 && unalign_pass > 0) ? 0 : 1;
}
