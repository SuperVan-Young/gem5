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
 * SEU MMIO test program for RISC-V
 *
 * This program tests the SpecializedExecutionUnit by issuing commands
 * via MMIO writes. It sends more commands than the queue depth to
 * naturally trigger timing backpressure.
 */

#define SEU_BASE 0x70000000
#define CMD_BYTES 16

/* Staging area: [SEU_BASE, SEU_BASE + CMD_BYTES) */
#define STAGING_ADDR SEU_BASE
/* Control area: [SEU_BASE + CMD_BYTES, SEU_BASE + 2*CMD_BYTES) */
#define CONTROL_ADDR (SEU_BASE + CMD_BYTES)

/* Volatile pointer for MMIO access */
volatile unsigned char *seu_staging = (volatile unsigned char *)STAGING_ADDR;
volatile unsigned int *seu_control = (volatile unsigned int *)CONTROL_ADDR;

static void write_cmd_to_staging(int cmd_id)
{
    int i;
    for (i = 0; i < CMD_BYTES; i++) {
        seu_staging[i] = (unsigned char)(cmd_id + i);
    }
}

static void launch_cmd(void)
{
    /* Writing 0 to control register triggers launch */
    *seu_control = 0;
}

int main(void)
{
    int num_cmds = 5;
    int i;

    /* Issue multiple commands, more than queue depth (2) */
    for (i = 0; i < num_cmds; i++) {
        write_cmd_to_staging(i);
        launch_cmd();
    }

    /* CPU exits, simulation will verify completed commands */
    return 0;
}
