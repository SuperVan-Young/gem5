#ifndef TESTS_GEM5_NPU_UTILS_CMD_VPU_H_
#define TESTS_GEM5_NPU_UTILS_CMD_VPU_H_

#include <stdint.h>

#include <vector>

#include "../npu_sync.hh"
#include "common.hh"

enum VpuOpcode
{
    VPU_OP_VADD = 0x1U,
    VPU_OP_VSUB = 0x2U,
    VPU_OP_VMUL = 0x3U,
    VPU_OP_VDIV = 0x4U,
    VPU_OP_VSCALE = 0x5U,
    VPU_OP_VCVT_I2F = 0x6U,
    VPU_OP_VCVT_F2I = 0x7U,
    VPU_OP_VSQRT = 0x8U,
    VPU_OP_VREDUCE_SUM = 0xAU,
    VPU_OP_VREDUCE_MAX = 0xBU,
    VPU_OP_VLOAD = 0xCU,
    VPU_OP_VSTORE = 0xDU,
    VPU_OP_VEXP = 0xEU,
};

enum VpuDataType
{
    VPU_DATA_I8 = 0x0U,
    VPU_DATA_I16 = 0x1U,
    VPU_DATA_I32 = 0x2U,
    VPU_DATA_U8 = 0x4U,
    VPU_DATA_U16 = 0x5U,
    VPU_DATA_U32 = 0x6U,
    VPU_DATA_F16 = 0x9U,
    VPU_DATA_F32 = 0xAU,
};

enum VpuLayoutOrder
{
    VPU_LAYOUT_WC = 0U,
    VPU_LAYOUT_CW = 1U,
};

enum VpuCmdWord
{
    VPU_CMD_WORD_FORMAT = 1U,
    VPU_CMD_WORD_DST_ADDR = 2U,
    VPU_CMD_WORD_DST_SHAPE = 3U,
    VPU_CMD_WORD_DST_STRIDE = 4U,
    VPU_CMD_WORD_SRC0_ADDR = 5U,
    VPU_CMD_WORD_SRC0_SHAPE = 6U,
    VPU_CMD_WORD_SRC0_STRIDE = 7U,
    VPU_CMD_WORD_SRC1_ADDR = 8U,
    VPU_CMD_WORD_SRC1_SHAPE = 9U,
    VPU_CMD_WORD_SRC1_STRIDE = 10U,
    VPU_CMD_WORD_EXTRA0 = 11U,
    VPU_CMD_WORD_EXTRA1 = 12U,
    VPU_CMD_WORD_EXTRA2 = 13U,
    VPU_CMD_WORD_EXTRA3 = 14U,
    VPU_CMD_WORD_EXTRA4 = 15U,
};

struct VpuCmdInstr
{
    uint32_t header;
    uint32_t format;
    uint32_t dst_addr;
    uint32_t dst_shape;
    uint32_t dst_stride;
    uint32_t src0_addr;
    uint32_t src0_shape;
    uint32_t src0_stride;
    uint32_t src1_addr;
    uint32_t src1_shape;
    uint32_t src1_stride;
    uint32_t extra0;
    uint32_t extra1;
    uint32_t extra2;
    uint32_t extra3;
    uint32_t extra4;
};

struct VpuTensorDesc
{
    uint32_t addr;
    uint32_t shape;
    uint32_t stride;
};

static_assert(sizeof(VpuCmdInstr) == NPU_CMD_BUFFER_BYTES,
              "VPU command struct must remain 64 bytes.");

enum VpuAddressLayout
{
    VPU_LOCAL_INPUT_BASE = 0x80000000U,
    VPU_LOCAL_OUTPUT_BASE = 0x81000000U,
    VPU_LOCAL_SLOT_STRIDE = 0x40U,
    VPU_DEFAULT_INPUT0_BUFFER = 0U,
    VPU_DEFAULT_INPUT1_BUFFER = 1U,
    VPU_DEFAULT_OUTPUT_BUFFER = 0U,
    VPU_DEFAULT_DLEN_BYTES = 4U,
};

static inline uint32_t
vpu_first_port(uint32_t mask)
{
    uint32_t port = 0U;
    while (((mask >> port) & 0x1U) == 0U) {
        ++port;
    }
    return port;
}

static inline uint32_t
vpu_local_addr(uint32_t base, uint32_t buffer_index)
{
    return base + (buffer_index * VPU_LOCAL_SLOT_STRIDE);
}

static inline uint32_t
vpu_dtype_size_bytes(uint32_t data_type)
{
    switch (data_type) {
      case VPU_DATA_I8:
      case VPU_DATA_U8:
        return 1U;
      case VPU_DATA_I16:
      case VPU_DATA_U16:
      case VPU_DATA_F16:
        return 2U;
      case VPU_DATA_I32:
      case VPU_DATA_U32:
      case VPU_DATA_F32:
        return 4U;
      default:
        return 4U;
    }
}

static inline uint32_t
vpu_fit_chunks(uint32_t value_minus_one)
{
    uint32_t chunks = 0U;
    while (value_minus_one != 0U && chunks < 7U) {
        ++chunks;
        value_minus_one >>= 4U;
    }
    return chunks;
}

static inline uint32_t
vpu_pack_axis_chunks(uint32_t mode, uint32_t w_value, uint32_t c_value)
{
    const uint32_t w_chunks = mode & 0x7U;
    const uint32_t c_chunks = 7U - w_chunks;
    uint32_t payload = 0U;

    if (w_chunks != 0U) {
        payload |=
            (w_value & ((1U << (w_chunks * 4U)) - 1U)) << (c_chunks * 4U);
    }
    if (c_chunks != 0U) {
        payload |= c_value & ((1U << (c_chunks * 4U)) - 1U);
    }
    return ((mode & 0xFU) << 28U) | payload;
}

static inline uint32_t
vpu_choose_axis_mode(uint32_t w_value, uint32_t c_value)
{
    const uint32_t min_w_chunks = vpu_fit_chunks(w_value);
    const uint32_t min_c_chunks = vpu_fit_chunks(c_value);
    uint32_t w_chunks = min_w_chunks;

    while (w_chunks <= 7U) {
        if ((7U - w_chunks) >= min_c_chunks) {
            return w_chunks;
        }
        ++w_chunks;
    }
    return 7U;
}

static inline uint32_t
vpu_pack_shape_field(uint32_t w_extent, uint32_t c_extent)
{
    const uint32_t w_value = w_extent == 0U ? 0U : (w_extent - 1U);
    const uint32_t c_value = c_extent == 0U ? 0U : (c_extent - 1U);
    const uint32_t mode = vpu_choose_axis_mode(w_value, c_value);
    return vpu_pack_axis_chunks(mode, w_value, c_value);
}

static inline uint32_t
vpu_pack_stride_field(uint32_t w_stride, uint32_t c_stride)
{
    const uint32_t w_value = w_stride == 0U ? 0U : (w_stride - 1U);
    const uint32_t c_value = c_stride == 0U ? 0U : (c_stride - 1U);
    const uint32_t mode = vpu_choose_axis_mode(w_value, c_value);
    return vpu_pack_axis_chunks(mode, w_value, c_value);
}

static inline uint32_t
vpu_pack_format_word(uint32_t dst_dtype, uint32_t src0_dtype,
                     uint32_t src1_dtype, uint32_t w_layout_log2,
                     uint32_t c_layout_log2, uint32_t layout_order)
{
    return (dst_dtype & 0xFU) |
        ((src0_dtype & 0xFU) << 4U) |
        ((src1_dtype & 0xFU) << 8U) |
        ((w_layout_log2 & 0xFU) << 12U) |
        ((c_layout_log2 & 0xFU) << 16U) |
        ((layout_order & 0xFU) << 20U);
}

static inline void
vpu_default_tensor_geometry(uint32_t elem_count, uint32_t data_type,
                            uint32_t *shape, uint32_t *stride,
                            uint32_t *w_layout_log2, uint32_t *c_layout_log2,
                            uint32_t *layout_order)
{
    const uint32_t elem_size = vpu_dtype_size_bytes(data_type);
    uint32_t lowest_dim = VPU_DEFAULT_DLEN_BYTES / elem_size;
    if (lowest_dim == 0U || (VPU_DEFAULT_DLEN_BYTES % elem_size) != 0U) {
        lowest_dim = 1U;
    }

    uint32_t c_extent = lowest_dim;
    if (elem_count == 0U) {
        elem_count = 1U;
    }
    if ((elem_count % c_extent) != 0U) {
        c_extent = 1U;
    }

    const uint32_t w_extent = elem_count / c_extent;
    *shape = vpu_pack_shape_field(w_extent, c_extent);
    *stride = vpu_pack_stride_field(1U, 1U);
    *w_layout_log2 = 0U;

    uint32_t log2_c = 0U;
    while ((1U << log2_c) < c_extent && log2_c < 15U) {
        ++log2_c;
    }
    *c_layout_log2 = log2_c;
    *layout_order = VPU_LAYOUT_WC;
}

static inline void
vpu_cmd_init_raw(NpuCmd *cmd, uint32_t device_id, uint32_t op_code,
                 uint32_t sync_indicator)
{
    VpuCmdInstr vpu_cmd = {};
    NpuCmdBinaryData binary = {};

    vpu_cmd.header = npuBuildHeaderWord(
        NPU_DEVICE_TYPE_VPU, device_id, op_code, sync_indicator,
        sync_indicator != 0U ? 1U : 0U, 0U);
    npuBinaryDataFromObject(&binary, vpu_cmd);
    cmd->loadBinary(binary);
}

static inline void
vpu_cmd_set_sync_indicator(NpuCmd *cmd, uint32_t sync_indicator)
{
    const uint32_t header = npuBuildHeaderWord(
        cmd->getDeviceType(),
        cmd->getDeviceId(),
        cmd->getOpCode(),
        sync_indicator,
        sync_indicator != 0U ? 1U : 0U,
        0U);
    cmd->setWord(0U, header);
}

static inline void
vpu_cmd_set_format(NpuCmd *cmd, uint32_t dst_dtype, uint32_t src0_dtype,
                   uint32_t src1_dtype, uint32_t w_layout_log2,
                   uint32_t c_layout_log2, uint32_t layout_order)
{
    NpuCmdBinaryData binary = {};
    VpuCmdInstr vpu_cmd = {};

    cmd->copyBinaryData(&binary);
    npuObjectFromBinaryData(&vpu_cmd, binary);
    vpu_cmd.format = vpu_pack_format_word(
        dst_dtype, src0_dtype, src1_dtype, w_layout_log2, c_layout_log2,
        layout_order);
    npuBinaryDataFromObject(&binary, vpu_cmd);
    cmd->loadBinary(binary);
}

static inline void
vpu_cmd_set_tensor(NpuCmd *cmd, uint32_t addr_word_index,
                   const VpuTensorDesc *desc)
{
    cmd->setWord(addr_word_index, desc->addr);
    cmd->setWord(addr_word_index + 1U, desc->shape);
    cmd->setWord(addr_word_index + 2U, desc->stride);
}

static inline void
vpu_cmd_set_extra_word(NpuCmd *cmd, uint32_t word_index, uint32_t value)
{
    cmd->setWord(word_index, value);
}

static inline void
vpu_cmd_init_compute(NpuCmd *cmd, uint32_t device_id, uint32_t op_code,
                     uint32_t sync_indicator, uint32_t dst_dtype,
                     uint32_t src0_dtype, uint32_t src1_dtype,
                     uint32_t w_layout_log2, uint32_t c_layout_log2,
                     uint32_t layout_order, const VpuTensorDesc *dst,
                     const VpuTensorDesc *src0, const VpuTensorDesc *src1,
                     uint32_t extra0)
{
    vpu_cmd_init_raw(cmd, device_id, op_code, sync_indicator);
    vpu_cmd_set_format(cmd, dst_dtype, src0_dtype, src1_dtype,
                       w_layout_log2, c_layout_log2, layout_order);
    vpu_cmd_set_tensor(cmd, VPU_CMD_WORD_DST_ADDR, dst);
    vpu_cmd_set_tensor(cmd, VPU_CMD_WORD_SRC0_ADDR, src0);
    vpu_cmd_set_tensor(cmd, VPU_CMD_WORD_SRC1_ADDR, src1);
    vpu_cmd_set_extra_word(cmd, VPU_CMD_WORD_EXTRA0, extra0);
}

static inline std::vector<NpuCmd>
vpu_cmd_make_unary_pipeline_template(uint32_t device_id, uint32_t op_code,
                                     uint32_t dst_dtype, uint32_t src0_dtype,
                                     uint32_t src1_dtype,
                                     uint32_t w_layout_log2,
                                     uint32_t c_layout_log2,
                                     uint32_t layout_order,
                                     const VpuTensorDesc *dst,
                                     const VpuTensorDesc *src0,
                                     const VpuTensorDesc *src1,
                                     const VpuTensorDesc *local_dst,
                                     const VpuTensorDesc *local_src0,
                                     uint32_t extra0)
{
    std::vector<NpuCmd> commands(3);
    vpu_cmd_init_compute(&commands[0], device_id, VPU_OP_VLOAD, 0U, src0_dtype,
                         src0_dtype, src0_dtype, w_layout_log2, c_layout_log2,
                         layout_order, local_src0, src0, src1, 0U);
    vpu_cmd_init_compute(&commands[1], device_id, op_code, 0U, dst_dtype,
                         src0_dtype, src1_dtype, w_layout_log2, c_layout_log2,
                         layout_order, local_dst, local_src0, local_src0, extra0);
    vpu_cmd_init_compute(&commands[2], device_id, VPU_OP_VSTORE, 0U, dst_dtype,
                         dst_dtype, dst_dtype, w_layout_log2, c_layout_log2,
                         layout_order, dst, local_dst, src1, 0U);
    return commands;
}

static inline std::vector<NpuCmd>
vpu_cmd_make_binary_pipeline_template(uint32_t device_id, uint32_t op_code,
                                      uint32_t dst_dtype,
                                      uint32_t src0_dtype,
                                      uint32_t src1_dtype,
                                      uint32_t w_layout_log2,
                                      uint32_t c_layout_log2,
                                      uint32_t layout_order,
                                      const VpuTensorDesc *dst,
                                      const VpuTensorDesc *src0,
                                      const VpuTensorDesc *src1,
                                      const VpuTensorDesc *local_dst,
                                      const VpuTensorDesc *local_src0,
                                      const VpuTensorDesc *local_src1)
{
    std::vector<NpuCmd> commands(4);
    const VpuTensorDesc zero = {0U, 0U, 0U};
    vpu_cmd_init_compute(&commands[0], device_id, VPU_OP_VLOAD, 0U,
                         src0_dtype, src0_dtype, src0_dtype, w_layout_log2,
                         c_layout_log2, layout_order, local_src0, src0, &zero,
                         0U);
    vpu_cmd_init_compute(&commands[1], device_id, VPU_OP_VLOAD, 0U,
                         src1_dtype, src1_dtype, src1_dtype, w_layout_log2,
                         c_layout_log2, layout_order, local_src1, src1, &zero,
                         0U);
    vpu_cmd_init_compute(&commands[2], device_id, op_code, 0U, dst_dtype,
                         src0_dtype, src1_dtype, w_layout_log2, c_layout_log2,
                         layout_order, local_dst, local_src0, local_src1, 0U);
    vpu_cmd_init_compute(&commands[3], device_id, VPU_OP_VSTORE, 0U, dst_dtype,
                         dst_dtype, dst_dtype, w_layout_log2, c_layout_log2,
                         layout_order, dst, local_dst, &zero, 0U);
    return commands;
}

static inline std::vector<NpuCmd>
vpu_cmd_make_reduce_pipeline_template(uint32_t device_id, uint32_t op_code,
                                      uint32_t dst_dtype,
                                      uint32_t src0_dtype,
                                      uint32_t src1_dtype,
                                      uint32_t w_layout_log2,
                                      uint32_t c_layout_log2,
                                      uint32_t layout_order,
                                      const VpuTensorDesc *dst,
                                      const VpuTensorDesc *src0,
                                      const VpuTensorDesc *src1,
                                      const VpuTensorDesc *local_dst,
                                      const VpuTensorDesc *local_src0)
{
    return vpu_cmd_make_unary_pipeline_template(
        device_id, op_code, dst_dtype, src0_dtype, src1_dtype, w_layout_log2,
        c_layout_log2, layout_order, dst, src0, src1, local_dst, local_src0,
        0U);
}

static inline void
vpu_cmd_launch_compute_at(uint64_t port_base, uint32_t device_id,
                          uint32_t op_code, uint32_t sync_indicator,
                          uint32_t dst_dtype, uint32_t src0_dtype,
                          uint32_t src1_dtype, uint32_t w_layout_log2,
                          uint32_t c_layout_log2, uint32_t layout_order,
                          const VpuTensorDesc *dst, const VpuTensorDesc *src0,
                          const VpuTensorDesc *src1, uint32_t extra0)
{
    NpuCmd cmd;
    vpu_cmd_init_compute(&cmd, device_id, op_code, sync_indicator, dst_dtype,
                         src0_dtype, src1_dtype, w_layout_log2, c_layout_log2,
                         layout_order, dst, src0, src1, extra0);
    cmd.launchCmdAt(port_base);
}

static inline void
vpu_cmd_launch_compute(uint32_t device_id, uint32_t op_code,
                       uint32_t sync_indicator, uint32_t dst_dtype,
                       uint32_t src0_dtype, uint32_t src1_dtype,
                       uint32_t w_layout_log2, uint32_t c_layout_log2,
                       uint32_t layout_order, const VpuTensorDesc *dst,
                       const VpuTensorDesc *src0, const VpuTensorDesc *src1,
                       uint32_t extra0)
{
    vpu_cmd_launch_compute_at(
        NPU_CMD_PORT_BASE, device_id, op_code, sync_indicator, dst_dtype,
        src0_dtype, src1_dtype, w_layout_log2, c_layout_log2, layout_order,
        dst, src0, src1, extra0);
}

static inline VpuTensorDesc
vpu_tensor_desc(uint32_t addr, uint32_t shape, uint32_t stride)
{
    VpuTensorDesc desc = {addr, shape, stride};
    return desc;
}

static inline VpuTensorDesc
vpu_default_local_tensor(uint32_t base, uint32_t buffer_index,
                         uint32_t elem_count, uint32_t data_type)
{
    uint32_t shape = 0U;
    uint32_t stride = 0U;
    uint32_t w_layout_log2 = 0U;
    uint32_t c_layout_log2 = 0U;
    uint32_t layout_order = 0U;
    (void)w_layout_log2;
    (void)c_layout_log2;
    (void)layout_order;
    vpu_default_tensor_geometry(elem_count, data_type, &shape, &stride,
                                &w_layout_log2, &c_layout_log2, &layout_order);
    return vpu_tensor_desc(vpu_local_addr(base, buffer_index), shape, stride);
}

static inline VpuTensorDesc
vpu_default_spm_tensor(uint32_t port, uint32_t elem_count, uint32_t data_type)
{
    uint32_t shape = 0U;
    uint32_t stride = 0U;
    uint32_t w_layout_log2 = 0U;
    uint32_t c_layout_log2 = 0U;
    uint32_t layout_order = 0U;
    (void)w_layout_log2;
    (void)c_layout_log2;
    (void)layout_order;
    vpu_default_tensor_geometry(elem_count, data_type, &shape, &stride,
                                &w_layout_log2, &c_layout_log2, &layout_order);
    return vpu_tensor_desc(0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE),
                           shape, stride);
}

static inline void
vpu_cmd_init_load_one(NpuCmd *cmd, uint32_t device_id, uint32_t sync_indicator,
                      uint32_t port, uint32_t elem_count,
                      uint32_t src_stride_bytes, uint32_t data_type,
                      uint32_t input_buffer_index)
{
    (void)src_stride_bytes;
    uint32_t w_layout_log2 = 0U;
    uint32_t c_layout_log2 = 0U;
    uint32_t layout_order = 0U;
    uint32_t shape = 0U;
    uint32_t stride = 0U;
    vpu_default_tensor_geometry(elem_count, data_type, &shape, &stride,
                                &w_layout_log2, &c_layout_log2, &layout_order);
    VpuTensorDesc src0 = vpu_tensor_desc(
        0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE), shape, stride);
    VpuTensorDesc dst = vpu_tensor_desc(
        vpu_local_addr(VPU_LOCAL_INPUT_BASE, input_buffer_index), shape, stride);
    const VpuTensorDesc src1 = {0U, 0U, 0U};
    vpu_cmd_init_compute(cmd, device_id, VPU_OP_VLOAD, sync_indicator,
                         data_type, data_type, data_type, w_layout_log2,
                         c_layout_log2, layout_order, &dst, &src0, &src1, 0U);
}

static inline void
vpu_cmd_launch_load_one_at(uint64_t port_base, uint32_t device_id,
                           uint32_t sync_indicator, uint32_t port,
                           uint32_t elem_count, uint32_t src_stride_bytes,
                           uint32_t data_type, uint32_t input_buffer_index)
{
    NpuCmd cmd;
    vpu_cmd_init_load_one(&cmd, device_id, sync_indicator, port, elem_count,
                          src_stride_bytes, data_type, input_buffer_index);
    cmd.launchCmdAt(port_base);
}

static inline void
vpu_cmd_launch_load_one(uint32_t device_id, uint32_t sync_indicator,
                        uint32_t port, uint32_t elem_count,
                        uint32_t src_stride_bytes, uint32_t data_type,
                        uint32_t input_buffer_index)
{
    vpu_cmd_launch_load_one_at(NPU_CMD_PORT_BASE, device_id, sync_indicator,
                               port, elem_count, src_stride_bytes, data_type,
                               input_buffer_index);
}

static inline void
vpu_cmd_init_store_one(NpuCmd *cmd, uint32_t device_id, uint32_t sync_indicator,
                       uint32_t port, uint32_t elem_count,
                       uint32_t dst_stride_bytes, uint32_t data_type,
                       uint32_t source_local_base, uint32_t buffer_index)
{
    (void)dst_stride_bytes;
    uint32_t w_layout_log2 = 0U;
    uint32_t c_layout_log2 = 0U;
    uint32_t layout_order = 0U;
    uint32_t shape = 0U;
    uint32_t stride = 0U;
    vpu_default_tensor_geometry(elem_count, data_type, &shape, &stride,
                                &w_layout_log2, &c_layout_log2, &layout_order);
    VpuTensorDesc src0 = vpu_tensor_desc(
        vpu_local_addr(source_local_base, buffer_index), shape, stride);
    VpuTensorDesc dst = vpu_tensor_desc(
        0x60000000U + (port * VPU_LOCAL_SLOT_STRIDE), shape, stride);
    const VpuTensorDesc src1 = {0U, 0U, 0U};
    vpu_cmd_init_compute(cmd, device_id, VPU_OP_VSTORE, sync_indicator,
                         data_type, data_type, data_type, w_layout_log2,
                         c_layout_log2, layout_order, &dst, &src0, &src1, 0U);
}

static inline void
vpu_cmd_launch_store_one_at(uint64_t port_base, uint32_t device_id,
                            uint32_t sync_indicator, uint32_t port,
                            uint32_t elem_count, uint32_t dst_stride_bytes,
                            uint32_t data_type, uint32_t source_local_base,
                            uint32_t buffer_index)
{
    NpuCmd cmd;
    vpu_cmd_init_store_one(&cmd, device_id, sync_indicator, port, elem_count,
                           dst_stride_bytes, data_type, source_local_base,
                           buffer_index);
    cmd.launchCmdAt(port_base);
}

static inline void
vpu_cmd_launch_store_one(uint32_t device_id, uint32_t sync_indicator,
                         uint32_t port, uint32_t elem_count,
                         uint32_t dst_stride_bytes, uint32_t data_type,
                         uint32_t source_local_base, uint32_t buffer_index)
{
    vpu_cmd_launch_store_one_at(
        NPU_CMD_PORT_BASE, device_id, sync_indicator, port, elem_count,
        dst_stride_bytes, data_type, source_local_base, buffer_index);
}

static inline void
vpu_cmd_launch_binary_at(uint64_t port_base, uint32_t device_id,
                         uint32_t op_code, uint32_t sync_indicator,
                         uint32_t read_mask, uint32_t write_mask,
                         uint32_t repetition, uint32_t elem_count,
                         uint32_t src_stride_bytes,
                         uint32_t dst_stride_bytes, uint32_t data_type)
{
    (void)src_stride_bytes;
    (void)dst_stride_bytes;

    for (uint32_t iter = 0U; iter < (repetition == 0U ? 1U : repetition);
         ++iter) {
        uint32_t reads = read_mask;
        uint32_t writes = write_mask;
        while (writes != 0U) {
            const uint32_t dst_port = vpu_first_port(writes);
            writes &= ~(1U << dst_port);
            const uint32_t src0_port = vpu_first_port(reads);
            reads &= ~(1U << src0_port);
            const uint32_t src1_port = vpu_first_port(reads);
            reads &= ~(1U << src1_port);

            vpu_cmd_launch_load_one_at(port_base, device_id, 0U, src0_port,
                                       elem_count, src_stride_bytes, data_type,
                                       VPU_DEFAULT_INPUT0_BUFFER);
            vpu_cmd_launch_load_one_at(port_base, device_id, 0U, src1_port,
                                       elem_count, src_stride_bytes, data_type,
                                       VPU_DEFAULT_INPUT1_BUFFER);

            uint32_t w_layout_log2 = 0U;
            uint32_t c_layout_log2 = 0U;
            uint32_t layout_order = 0U;
            uint32_t shape = 0U;
            uint32_t stride = 0U;
            vpu_default_tensor_geometry(elem_count, data_type, &shape, &stride,
                                        &w_layout_log2, &c_layout_log2,
                                        &layout_order);
            const VpuTensorDesc dst = vpu_tensor_desc(
                vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER),
                shape, stride);
            const VpuTensorDesc src0 = vpu_tensor_desc(
                vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT0_BUFFER),
                shape, stride);
            const VpuTensorDesc src1 = vpu_tensor_desc(
                vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT1_BUFFER),
                shape, stride);
            vpu_cmd_launch_compute_at(
                port_base, device_id, op_code, 0U, data_type, data_type,
                data_type, w_layout_log2, c_layout_log2, layout_order, &dst,
                &src0, &src1, 0U);

            const uint32_t store_sync =
                (iter + 1U == (repetition == 0U ? 1U : repetition) &&
                 writes == 0U) ?
                sync_indicator : 0U;
            vpu_cmd_launch_store_one_at(
                port_base, device_id, store_sync, dst_port, elem_count,
                dst_stride_bytes, data_type, VPU_LOCAL_OUTPUT_BASE,
                VPU_DEFAULT_OUTPUT_BUFFER);
        }
    }
}

static inline void
vpu_cmd_launch_binary(uint32_t device_id, uint32_t op_code,
                      uint32_t sync_indicator, uint32_t read_mask,
                      uint32_t write_mask, uint32_t repetition,
                      uint32_t elem_count, uint32_t src_stride_bytes,
                      uint32_t dst_stride_bytes, uint32_t data_type)
{
    vpu_cmd_launch_binary_at(NPU_CMD_PORT_BASE, device_id, op_code,
                             sync_indicator, read_mask, write_mask, repetition,
                             elem_count, src_stride_bytes, dst_stride_bytes,
                             data_type);
}

static inline void
vpu_cmd_launch_unary_at(uint64_t port_base, uint32_t device_id,
                        uint32_t op_code, uint32_t sync_indicator,
                        uint32_t read_mask, uint32_t write_mask,
                        uint32_t repetition, uint32_t elem_count,
                        uint32_t src_stride_bytes,
                        uint32_t dst_stride_bytes, uint32_t data_type)
{
    (void)src_stride_bytes;
    (void)dst_stride_bytes;

    const uint32_t store_elem_count =
        (op_code == VPU_OP_VREDUCE_SUM || op_code == VPU_OP_VREDUCE_MAX) ?
        1U : elem_count;

    for (uint32_t iter = 0U; iter < (repetition == 0U ? 1U : repetition);
         ++iter) {
        uint32_t reads = read_mask;
        uint32_t writes = write_mask;
        while (writes != 0U) {
            const uint32_t dst_port = vpu_first_port(writes);
            writes &= ~(1U << dst_port);
            const uint32_t src_port = vpu_first_port(reads);
            reads &= ~(1U << src_port);

            vpu_cmd_launch_load_one_at(port_base, device_id, 0U, src_port,
                                       elem_count, src_stride_bytes, data_type,
                                       VPU_DEFAULT_INPUT0_BUFFER);

            uint32_t w_layout_log2 = 0U;
            uint32_t c_layout_log2 = 0U;
            uint32_t layout_order = 0U;
            uint32_t shape = 0U;
            uint32_t stride = 0U;
            vpu_default_tensor_geometry(elem_count, data_type, &shape, &stride,
                                        &w_layout_log2, &c_layout_log2,
                                        &layout_order);
            const VpuTensorDesc dst = vpu_tensor_desc(
                vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER),
                op_code == VPU_OP_VREDUCE_SUM || op_code == VPU_OP_VREDUCE_MAX ?
                vpu_pack_shape_field(1U, 1U) : shape,
                stride);
            const VpuTensorDesc src0 = vpu_tensor_desc(
                vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT0_BUFFER),
                shape, stride);
            const VpuTensorDesc src1 = {0U, 0U, 0U};
            vpu_cmd_launch_compute_at(
                port_base, device_id, op_code, 0U, data_type, data_type,
                data_type, w_layout_log2, c_layout_log2, layout_order, &dst,
                &src0, &src1, 0U);

            const uint32_t store_sync =
                (iter + 1U == (repetition == 0U ? 1U : repetition) &&
                 writes == 0U) ?
                sync_indicator : 0U;
            vpu_cmd_launch_store_one_at(
                port_base, device_id, store_sync, dst_port, store_elem_count,
                dst_stride_bytes, data_type, VPU_LOCAL_OUTPUT_BASE,
                VPU_DEFAULT_OUTPUT_BUFFER);
        }
    }
}

static inline void
vpu_cmd_launch_unary(uint32_t device_id, uint32_t op_code,
                     uint32_t sync_indicator, uint32_t read_mask,
                     uint32_t write_mask, uint32_t repetition,
                     uint32_t elem_count, uint32_t src_stride_bytes,
                     uint32_t dst_stride_bytes, uint32_t data_type)
{
    vpu_cmd_launch_unary_at(NPU_CMD_PORT_BASE, device_id, op_code,
                            sync_indicator, read_mask, write_mask, repetition,
                            elem_count, src_stride_bytes, dst_stride_bytes,
                            data_type);
}

static inline void
vpu_cmd_launch_scale_at(uint64_t port_base, uint32_t device_id,
                        uint32_t sync_indicator, uint32_t read_mask,
                        uint32_t write_mask, uint32_t repetition,
                        uint32_t elem_count, uint32_t src_stride_bytes,
                        uint32_t dst_stride_bytes, uint32_t data_type,
                        uint32_t scalar_bits)
{
    (void)src_stride_bytes;
    (void)dst_stride_bytes;

    for (uint32_t iter = 0U; iter < (repetition == 0U ? 1U : repetition);
         ++iter) {
        uint32_t reads = read_mask;
        uint32_t writes = write_mask;
        while (writes != 0U) {
            const uint32_t dst_port = vpu_first_port(writes);
            writes &= ~(1U << dst_port);
            const uint32_t src_port = vpu_first_port(reads);
            reads &= ~(1U << src_port);

            vpu_cmd_launch_load_one_at(port_base, device_id, 0U, src_port,
                                       elem_count, src_stride_bytes, data_type,
                                       VPU_DEFAULT_INPUT0_BUFFER);

            uint32_t w_layout_log2 = 0U;
            uint32_t c_layout_log2 = 0U;
            uint32_t layout_order = 0U;
            uint32_t shape = 0U;
            uint32_t stride = 0U;
            vpu_default_tensor_geometry(elem_count, data_type, &shape, &stride,
                                        &w_layout_log2, &c_layout_log2,
                                        &layout_order);
            const VpuTensorDesc dst = vpu_tensor_desc(
                vpu_local_addr(VPU_LOCAL_OUTPUT_BASE, VPU_DEFAULT_OUTPUT_BUFFER),
                shape, stride);
            const VpuTensorDesc src0 = vpu_tensor_desc(
                vpu_local_addr(VPU_LOCAL_INPUT_BASE, VPU_DEFAULT_INPUT0_BUFFER),
                shape, stride);
            const VpuTensorDesc src1 = {0U, 0U, 0U};
            vpu_cmd_launch_compute_at(
                port_base, device_id, VPU_OP_VSCALE, 0U, data_type, data_type,
                data_type, w_layout_log2, c_layout_log2, layout_order, &dst,
                &src0, &src1, scalar_bits);

            const uint32_t store_sync =
                (iter + 1U == (repetition == 0U ? 1U : repetition) &&
                 writes == 0U) ?
                sync_indicator : 0U;
            vpu_cmd_launch_store_one_at(
                port_base, device_id, store_sync, dst_port, elem_count,
                dst_stride_bytes, data_type, VPU_LOCAL_OUTPUT_BASE,
                VPU_DEFAULT_OUTPUT_BUFFER);
        }
    }
}

static inline void
vpu_cmd_launch_scale(uint32_t device_id, uint32_t sync_indicator,
                     uint32_t read_mask, uint32_t write_mask,
                     uint32_t repetition, uint32_t elem_count,
                     uint32_t src_stride_bytes, uint32_t dst_stride_bytes,
                     uint32_t data_type, uint32_t scalar_bits)
{
    vpu_cmd_launch_scale_at(NPU_CMD_PORT_BASE, device_id, sync_indicator,
                            read_mask, write_mask, repetition, elem_count,
                            src_stride_bytes, dst_stride_bytes, data_type,
                            scalar_bits);
}

static inline void
vpu_cmd_launch_load_at(uint64_t port_base, uint32_t device_id,
                       uint32_t sync_indicator, uint32_t read_mask,
                       uint32_t elem_count, uint32_t src_stride_bytes,
                       uint32_t data_type)
{
    uint32_t reads = read_mask;
    uint32_t buffer_index = 0U;
    while (reads != 0U) {
        const uint32_t port = vpu_first_port(reads);
        reads &= ~(1U << port);
        const uint32_t load_sync = reads == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_load_one_at(port_base, device_id, load_sync, port,
                                   elem_count, src_stride_bytes, data_type,
                                   buffer_index++);
    }
}

static inline void
vpu_cmd_launch_load(uint32_t device_id, uint32_t sync_indicator,
                    uint32_t read_mask, uint32_t elem_count,
                    uint32_t src_stride_bytes, uint32_t data_type)
{
    vpu_cmd_launch_load_at(NPU_CMD_PORT_BASE, device_id, sync_indicator,
                           read_mask, elem_count, src_stride_bytes, data_type);
}

static inline void
vpu_cmd_launch_store_at(uint64_t port_base, uint32_t device_id,
                        uint32_t sync_indicator, uint32_t write_mask,
                        uint32_t elem_count, uint32_t dst_stride_bytes,
                        uint32_t data_type)
{
    uint32_t writes = write_mask;
    uint32_t buffer_index = 0U;
    while (writes != 0U) {
        const uint32_t port = vpu_first_port(writes);
        writes &= ~(1U << port);
        const uint32_t store_sync = writes == 0U ? sync_indicator : 0U;
        vpu_cmd_launch_store_one_at(port_base, device_id, store_sync, port,
                                    elem_count, dst_stride_bytes, data_type,
                                    VPU_LOCAL_INPUT_BASE, buffer_index++);
    }
}

static inline void
vpu_cmd_launch_store(uint32_t device_id, uint32_t sync_indicator,
                     uint32_t write_mask, uint32_t elem_count,
                     uint32_t dst_stride_bytes, uint32_t data_type)
{
    vpu_cmd_launch_store_at(NPU_CMD_PORT_BASE, device_id, sync_indicator,
                            write_mask, elem_count, dst_stride_bytes,
                            data_type);
}

#endif
