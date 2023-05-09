// Copyright 2023 quocdang1998
#ifndef MERLIN_GRADIENT_DESCENT_HPP_
#define MERLIN_GRADIENT_DESCENT_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/cuda_decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec

namespace merlin::candy {

/** @brief Convert contiguous index to ndim index with a dimension fixed.*/
__cuhostdev__ intvec contiguous_to_ndim_idx_1(std::uint64_t index, const intvec & shape, std::uint64_t skip_dim,
                                              std::uint64_t * data_ptr = nullptr);

/** @brief Parallel calculate gradient of a model over dataset by CPU.
 *  @param model Canonical decomposition model.
 *  @param train_data Target data to fit the model.
 */
MERLIN_EXPORTS Vector<double> calc_gradient_vector_cpu(const candy::Model & model, const array::Array & train_data);

#ifdef __NVCC__

/** @brief Calculate gradient of a model over dataset on GPU.
 *  @param p_model Pointer to canonical decomposition model.
 *  @param p_train_data Pointer to target data to fit the model.
 *  @param cache_memory Pointer to cache memory for calculation (must be able to hold model shape vector and index
 *  vectors of each thread).
 *  @param gradient_vector Pointer to memory where the gradient is stored.
 */
__cudevice__ void calc_gradient_vector_gpu(const candy::Model * p_model, const array::Parcel * p_train_data,
                                           std::uint64_t * cache_memory, double * gradient_vector);

#endif  // __NVCC__

}  // namespace merlin::candy

#endif  // MERLIN_GRADIENT_DESCENT_HPP_
