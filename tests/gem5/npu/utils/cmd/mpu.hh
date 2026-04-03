#ifndef TESTS_GEM5_NPU_UTILS_CMD_MPU_H_
#define TESTS_GEM5_NPU_UTILS_CMD_MPU_H_

#include <stdint.h>

#include "common.hh"

enum MpuDataType
{
    MPU_DATA_TYPE_INT8 = 0x0U,
};

enum MpuMode
{
    MPU_MODE_LOAD = 0x0U,
    MPU_MODE_COMPUTE = 0x1U,
    MPU_MODE_STORE = 0x2U,
    MPU_MODE_TENSOR_LOOP = 0x4U,
};

enum MpuComputeSubop
{
    MPU_SUBOP_MATMUL = 0x0U,
    MPU_SUBOP_MATMUL_ACC = 0x1U,
};

enum MpuAxis
{
    MPU_AXIS_M = 0x0U,
    MPU_AXIS_N = 0x1U,
    MPU_AXIS_K = 0x2U,
};

enum MpuLocalAddr
{
    MPU_LOCAL_ADDR_A0 = 0x00000100U,
    MPU_LOCAL_ADDR_A1 = 0x00000110U,
    MPU_LOCAL_ADDR_B0 = 0x00000120U,
    MPU_LOCAL_ADDR_B1 = 0x00000130U,
    MPU_LOCAL_ADDR_C0 = 0x00000140U,
    MPU_LOCAL_ADDR_C1 = 0x00000150U,
};

enum MpuLayoutMode
{
    MPU_LAYOUT_MODE_NORMAL = 0x0U,
    MPU_LAYOUT_MODE_SKEWED = 0x1U,
};

static inline uint32_t
mpuAlternateLocalAddr(uint32_t local_addr)
{
    switch (local_addr) {
      case MPU_LOCAL_ADDR_A0:
        return MPU_LOCAL_ADDR_A1;
      case MPU_LOCAL_ADDR_A1:
        return MPU_LOCAL_ADDR_A0;
      case MPU_LOCAL_ADDR_B0:
        return MPU_LOCAL_ADDR_B1;
      case MPU_LOCAL_ADDR_B1:
        return MPU_LOCAL_ADDR_B0;
      case MPU_LOCAL_ADDR_C0:
        return MPU_LOCAL_ADDR_C1;
      case MPU_LOCAL_ADDR_C1:
        return MPU_LOCAL_ADDR_C0;
      default:
        return 0U;
    }
}

static inline void
mpuResolveTensorLoopSlotPair(uint32_t base_local_addr,
                             uint32_t pingpong_enable,
                             uint32_t *primary_local_addr,
                             uint32_t *alternate_local_addr)
{
    if (primary_local_addr != nullptr) {
        *primary_local_addr = base_local_addr;
    }

    if (alternate_local_addr != nullptr) {
        *alternate_local_addr =
            pingpong_enable ? mpuAlternateLocalAddr(base_local_addr)
                            : base_local_addr;
    }
}

static inline uint32_t
mpuMakeOpCode(uint32_t data_type, uint32_t mode)
{
    return ((data_type & 0x7U) << 5) | ((mode & 0x7U) << 2);
}

static inline uint32_t
mpuBuildLoadStoreCtrl(uint32_t layout_mode, uint32_t local_offset_bytes)
{
    return (layout_mode & 0x3U) | ((local_offset_bytes & 0xffU) << 8);
}

static inline uint32_t
mpuBuildLoopCtrl0(uint32_t subop, uint32_t layout_a, uint32_t layout_b,
                  uint32_t layout_c, uint32_t outer_axis,
                  uint32_t inner_axis)
{
    return (subop & 0x3U) |
           ((layout_a & 0x3U) << 4) |
           ((layout_b & 0x3U) << 8) |
           ((layout_c & 0x3U) << 12) |
           ((outer_axis & 0x3U) << 16) |
           ((inner_axis & 0x3U) << 20);
}

static inline uint32_t
mpuBuildOffsetPack(uint32_t offset_a, uint32_t offset_b, uint32_t offset_c)
{
    return (offset_a & 0xffU) |
           ((offset_b & 0xffU) << 8) |
           ((offset_c & 0xffU) << 16);
}

static inline uint32_t
mpuBuildStepCfg(uint32_t outer_step_tiles, uint32_t inner_step_tiles,
                uint32_t pingpong_a_enable, uint32_t pingpong_b_enable,
                uint32_t pingpong_c_enable, uint32_t auto_load_c_for_acc)
{
    return (outer_step_tiles & 0xffU) |
           ((inner_step_tiles & 0xffU) << 8) |
           ((pingpong_a_enable & 0x1U) << 16) |
           ((pingpong_b_enable & 0x1U) << 17) |
           ((pingpong_c_enable & 0x1U) << 18) |
           ((auto_load_c_for_acc & 0x1U) << 19);
}

static inline void
npuBuildMpuLoadCmd(NpuCmd *cmd, uint32_t device_id,
                   uint32_t dst_local_addr, uint32_t src_spm_addr,
                   uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                   uint32_t layout_mode, uint32_t local_offset_bytes,
                   uint32_t sync_indicator, uint32_t set_completion_sync)
{
    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_MPU);
    cmd->setDeviceId(device_id);
    cmd->setOpCode(mpuMakeOpCode(MPU_DATA_TYPE_INT8, MPU_MODE_LOAD));
    cmd->setSyncIndicator(sync_indicator);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setSetIndicatorSnd(0U);
    cmd->clearCommonReservedBits();
    cmd->setWord(1U, dst_local_addr);
    cmd->setWord(2U, src_spm_addr);
    cmd->setWord(3U, valid_m);
    cmd->setWord(4U, valid_n);
    cmd->setWord(5U, valid_k);
    cmd->setWord(6U, mpuBuildLoadStoreCtrl(layout_mode, local_offset_bytes));
}

static inline void
npuBuildMpuComputeCmd(NpuCmd *cmd, uint32_t device_id,
                      uint32_t src_local_addr_a, uint32_t src_local_addr_b,
                      uint32_t dst_local_addr_c, uint32_t valid_m,
                      uint32_t valid_n, uint32_t valid_k, uint32_t subop,
                      uint32_t dst_layout_mode, uint32_t compute_mode_flags,
                      uint32_t local_offset_a, uint32_t local_offset_b,
                      uint32_t local_offset_c, uint32_t sync_indicator,
                      uint32_t set_completion_sync)
{
    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_MPU);
    cmd->setDeviceId(device_id);
    cmd->setOpCode(mpuMakeOpCode(MPU_DATA_TYPE_INT8, MPU_MODE_COMPUTE));
    cmd->setSyncIndicator(sync_indicator);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setSetIndicatorSnd(0U);
    cmd->clearCommonReservedBits();
    cmd->setWord(1U, src_local_addr_a);
    cmd->setWord(2U, src_local_addr_b);
    cmd->setWord(3U, dst_local_addr_c);
    cmd->setWord(4U, valid_m);
    cmd->setWord(5U, valid_n);
    cmd->setWord(6U, valid_k);
    cmd->setWord(7U, subop);
    cmd->setWord(8U, dst_layout_mode);
    cmd->setWord(9U, compute_mode_flags);
    cmd->setWord(10U, local_offset_a);
    cmd->setWord(11U, local_offset_b);
    cmd->setWord(12U, local_offset_c);
}

static inline void
npuBuildMpuStoreCmd(NpuCmd *cmd, uint32_t device_id,
                    uint32_t src_local_addr, uint32_t dst_spm_addr,
                    uint32_t valid_m, uint32_t valid_n, uint32_t valid_k,
                    uint32_t layout_mode, uint32_t local_offset_bytes,
                    uint32_t sync_indicator, uint32_t set_completion_sync)
{
    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_MPU);
    cmd->setDeviceId(device_id);
    cmd->setOpCode(mpuMakeOpCode(MPU_DATA_TYPE_INT8, MPU_MODE_STORE));
    cmd->setSyncIndicator(sync_indicator);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setSetIndicatorSnd(0U);
    cmd->clearCommonReservedBits();
    cmd->setWord(1U, src_local_addr);
    cmd->setWord(2U, dst_spm_addr);
    cmd->setWord(3U, valid_m);
    cmd->setWord(4U, valid_n);
    cmd->setWord(5U, valid_k);
    cmd->setWord(6U, mpuBuildLoadStoreCtrl(layout_mode, local_offset_bytes));
}

static inline void
npuBuildMpuTensorLoopCmd(NpuCmd *cmd, uint32_t device_id,
                         uint32_t base_local_addr_a,
                         uint32_t base_local_addr_b,
                         uint32_t base_local_addr_c,
                         uint32_t base_spm_addr_a,
                         uint32_t base_spm_addr_b,
                         uint32_t base_spm_addr_c, uint32_t valid_m,
                         uint32_t valid_n, uint32_t valid_k, uint32_t loop_ctrl0,
                         uint32_t outer_count, uint32_t inner_count,
                         uint32_t offset_pack, uint32_t step_cfg,
                         uint32_t sync_indicator,
                         uint32_t set_completion_sync)
{
    cmd->clear();
    cmd->setDeviceType(NPU_DEVICE_TYPE_MPU);
    cmd->setDeviceId(device_id);
    cmd->setOpCode(mpuMakeOpCode(MPU_DATA_TYPE_INT8, MPU_MODE_TENSOR_LOOP));
    cmd->setSyncIndicator(sync_indicator);
    cmd->setSetIndicatorSns(set_completion_sync ? 1U : 0U);
    cmd->setSetIndicatorSnd(0U);
    cmd->clearCommonReservedBits();
    cmd->setWord(1U, base_local_addr_a);
    cmd->setWord(2U, base_local_addr_b);
    cmd->setWord(3U, base_local_addr_c);
    cmd->setWord(4U, base_spm_addr_a);
    cmd->setWord(5U, base_spm_addr_b);
    cmd->setWord(6U, base_spm_addr_c);
    cmd->setWord(7U, valid_m);
    cmd->setWord(8U, valid_n);
    cmd->setWord(9U, valid_k);
    cmd->setWord(10U, loop_ctrl0);
    cmd->setWord(11U, outer_count);
    cmd->setWord(12U, inner_count);
    cmd->setWord(13U, offset_pack);
    cmd->setWord(14U, step_cfg);
    cmd->setWord(15U, 0U);
}

#endif
