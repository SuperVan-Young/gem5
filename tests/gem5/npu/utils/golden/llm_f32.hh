#ifndef TESTS_GEM5_NPU_UTILS_GOLDEN_LLM_F32_HH_
#define TESTS_GEM5_NPU_UTILS_GOLDEN_LLM_F32_HH_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void npu_golden_rmsnorm_lastdim_f32(const uint32_t *src_bits,
                                    const uint32_t *weight_bits, uint32_t rows,
                                    uint32_t cols, float epsilon,
                                    uint32_t *dst_bits);
void npu_golden_softmax_lastdim_f32(const uint32_t *src_bits, uint32_t rows,
                                    uint32_t cols, uint32_t *dst_bits);
void npu_golden_swiglu_lastdim_f32(const uint32_t *gate_bits,
                                   const uint32_t *value_bits, uint32_t rows,
                                   uint32_t cols, uint32_t *dst_bits);

/*
 * FlashAttention Stage 1 golden contract:
 * - all tensors are row-major logical matrices serialized as float32 bits
 * - Q: [seq_q, dim]
 * - K: [seq_k, dim]
 * - V: [seq_k, dim_v]
 * - scores/P: [seq_q, seq_k]
 * - O: [seq_q, dim_v]
 * - scaling is applied after Q*K^T and before row-wise softmax
 */
float npu_golden_flashattention_default_scale_f32(uint32_t dim);
void npu_golden_qk_scores_f32(const uint32_t *q_bits, const uint32_t *k_bits,
                              uint32_t seq_q, uint32_t seq_k, uint32_t dim,
                              float scaling, uint32_t *scores_bits);
void npu_golden_attention_probs_f32(const uint32_t *scores_bits,
                                    uint32_t seq_q, uint32_t seq_k,
                                    uint32_t *prob_bits);
void npu_golden_flashattention_f32(const uint32_t *q_bits,
                                   const uint32_t *k_bits,
                                   const uint32_t *v_bits, uint32_t seq_q,
                                   uint32_t seq_k, uint32_t dim,
                                   uint32_t dim_v, float scaling,
                                   uint32_t *scores_bits, uint32_t *prob_bits,
                                   uint32_t *dst_bits);

#ifdef __cplusplus
}
#endif

#endif
