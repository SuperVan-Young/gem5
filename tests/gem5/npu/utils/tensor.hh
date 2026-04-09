#ifndef TESTS_GEM5_NPU_UTILS_TENSOR_HH_
#define TESTS_GEM5_NPU_UTILS_TENSOR_HH_

#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include "cmd/vpu.hh"

struct PrimitiveVpu2DDesc
{
    VpuTensorDesc tensor;
    uint32_t dataType = VPU_DATA_F32;
    uint32_t wElems = 1U;
    uint32_t cElems = 1U;
    uint32_t wLayoutElems = 1U;
    uint32_t cLayoutElems = 1U;
    uint32_t wLayoutLog2 = 0U;
    uint32_t cLayoutLog2 = 0U;
    uint32_t layoutOrder = VPU_LAYOUT_WC;
};

struct PrimitiveTensorDesc
{
    uint32_t baseAddr = 0U;
    uint32_t dataType = VPU_DATA_F32;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> strideElems;

    static PrimitiveTensorDesc
    dense(uint32_t base_addr, uint32_t data_type,
          std::initializer_list<uint32_t> dims)
    {
        PrimitiveTensorDesc desc;
        desc.baseAddr = base_addr;
        desc.dataType = data_type;
        desc.shape.assign(dims.begin(), dims.end());
        desc.strideElems.resize(desc.shape.size(), 1U);
        uint32_t running = 1U;
        for (size_t i = desc.shape.size(); i > 0U; --i) {
            desc.strideElems[i - 1U] = running;
            running *= desc.shape[i - 1U];
        }
        return desc;
    }

    static PrimitiveTensorDesc
    denseSpm(uint32_t slot, uint32_t data_type,
             std::initializer_list<uint32_t> dims)
    {
        return dense(0x60000000U + (slot * VPU_LOCAL_SLOT_STRIDE), data_type,
                     dims);
    }

    uint32_t
    elemBytes() const
    {
        return vpu_dtype_size_bytes(dataType);
    }

    size_t
    rank() const
    {
        return shape.size();
    }

    uint32_t
    dim(size_t axis) const
    {
        return shape[axis];
    }

    uint64_t
    numel() const
    {
        uint64_t count = 1U;
        for (uint32_t dim_size : shape) {
            count *= dim_size;
        }
        return count;
    }

    uint64_t
    bytes() const
    {
        return numel() * elemBytes();
    }

    uint64_t
    sliceCount(size_t keep_dims = 2U) const
    {
        if (keep_dims == 0U || keep_dims > shape.size()) {
            return 1U;
        }
        uint64_t count = 1U;
        for (size_t axis = 0U; axis + keep_dims < shape.size(); ++axis) {
            count *= shape[axis];
        }
        return count;
    }

    PrimitiveTensorDesc
    permute(std::initializer_list<uint32_t> order) const
    {
        return permute(std::vector<uint32_t>(order.begin(), order.end()));
    }

    PrimitiveTensorDesc
    permute(const std::vector<uint32_t> &order) const
    {
        PrimitiveTensorDesc out = *this;
        out.shape.clear();
        out.strideElems.clear();
        for (uint32_t axis : order) {
            if (axis >= shape.size()) {
                __builtin_trap();
            }
            out.shape.push_back(shape[axis]);
            out.strideElems.push_back(strideElems[axis]);
        }
        if (out.shape.size() != shape.size()) {
            __builtin_trap();
        }
        return out;
    }

    PrimitiveTensorDesc
    mergeDims(size_t begin, size_t end) const
    {
        if (begin >= end || end > shape.size()) {
            __builtin_trap();
        }
        for (size_t axis = begin; axis + 1U < end; ++axis) {
            if (strideElems[axis] != shape[axis + 1U] * strideElems[axis + 1U]) {
                __builtin_trap();
            }
        }

        PrimitiveTensorDesc out;
        out.baseAddr = baseAddr;
        out.dataType = dataType;
        out.shape.insert(out.shape.end(), shape.begin(), shape.begin() + begin);
        out.strideElems.insert(
            out.strideElems.end(), strideElems.begin(), strideElems.begin() + begin);

        uint32_t merged = 1U;
        for (size_t axis = begin; axis < end; ++axis) {
            merged *= shape[axis];
        }
        out.shape.push_back(merged);
        out.strideElems.push_back(strideElems[end - 1U]);
        out.shape.insert(out.shape.end(), shape.begin() + end, shape.end());
        out.strideElems.insert(
            out.strideElems.end(), strideElems.begin() + end, strideElems.end());
        return out;
    }

    PrimitiveTensorDesc
    peel(size_t linear_index, size_t keep_dims = 2U) const
    {
        if (keep_dims == 0U || keep_dims > shape.size()) {
            __builtin_trap();
        }
        const size_t peeled = shape.size() - keep_dims;
        uint64_t offset_elems = 0U;
        size_t rem = linear_index;
        for (size_t axis = 0U; axis < peeled; ++axis) {
            uint32_t suffix = 1U;
            for (size_t next = axis + 1U; next < peeled; ++next) {
                suffix *= shape[next];
            }
            const uint32_t idx = suffix == 0U ? 0U : rem / suffix;
            rem = suffix == 0U ? 0U : rem % suffix;
            if (idx >= shape[axis]) {
                __builtin_trap();
            }
            offset_elems += static_cast<uint64_t>(idx) * strideElems[axis];
        }

        PrimitiveTensorDesc out;
        out.baseAddr = baseAddr + (offset_elems * elemBytes());
        out.dataType = dataType;
        out.shape.insert(out.shape.end(), shape.end() - keep_dims, shape.end());
        out.strideElems.insert(
            out.strideElems.end(), strideElems.end() - keep_dims, strideElems.end());
        return out;
    }

    PrimitiveTensorDesc
    asWVector2D() const
    {
        if (shape.size() != 1U || strideElems[0] != 1U) {
            __builtin_trap();
        }
        PrimitiveTensorDesc out;
        out.baseAddr = baseAddr;
        out.dataType = dataType;
        out.shape = {shape[0], 1U};
        out.strideElems = {1U, shape[0]};
        return out;
    }

    PrimitiveTensorDesc
    asScalar2D() const
    {
        if (numel() != 1U) {
            __builtin_trap();
        }
        PrimitiveTensorDesc out;
        out.baseAddr = baseAddr;
        out.dataType = dataType;
        out.shape = {1U, 1U};
        out.strideElems = {1U, 1U};
        return out;
    }

    PrimitiveVpu2DDesc
    lowerToVpu2D(uint32_t w_layout_elems, uint32_t c_layout_elems,
                 uint32_t layout_order) const
    {
        if (shape.size() != 2U || strideElems.size() != 2U) {
            __builtin_trap();
        }
        if (w_layout_elems == 0U || c_layout_elems == 0U) {
            __builtin_trap();
        }
        if ((w_layout_elems & (w_layout_elems - 1U)) != 0U ||
            (c_layout_elems & (c_layout_elems - 1U)) != 0U) {
            __builtin_trap();
        }
        if (shape[0] > 1U && (shape[0] % w_layout_elems) != 0U) {
            __builtin_trap();
        }
        if (shape[1] > 1U && (shape[1] % c_layout_elems) != 0U) {
            __builtin_trap();
        }
        if ((shape[0] > 1U && (strideElems[0] % c_layout_elems) != 0U) ||
            (shape[1] > 1U && (strideElems[1] % w_layout_elems) != 0U)) {
            __builtin_trap();
        }

        PrimitiveVpu2DDesc out;
        out.dataType = dataType;
        out.wElems = shape[0];
        out.cElems = shape[1];
        out.wLayoutElems = w_layout_elems;
        out.cLayoutElems = c_layout_elems;
        out.layoutOrder = layout_order;
        out.tensor.addr = baseAddr;
        out.tensor.shape = vpu_pack_shape_field(shape[0], shape[1]);
        const uint32_t w_stride_tiles =
            shape[0] == 1U ? 1U : (strideElems[0] / c_layout_elems);
        const uint32_t c_stride_tiles =
            shape[1] == 1U ? 1U : (strideElems[1] / w_layout_elems);
        out.tensor.stride =
            vpu_pack_stride_field(w_stride_tiles, c_stride_tiles);
        while ((1U << out.wLayoutLog2) < w_layout_elems) {
            ++out.wLayoutLog2;
        }
        while ((1U << out.cLayoutLog2) < c_layout_elems) {
            ++out.cLayoutLog2;
        }
        return out;
    }
};

#endif
