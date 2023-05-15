// Copyright 2023 quocdang1998
#ifndef MERLIN_GRADIENT_DESCENT_HPP_
#define MERLIN_GRADIENT_DESCENT_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/cuda_decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::Vector, merlin::intvec, merlin::floatvec

namespace merlin::candy {

// GPU kernel wrapper
// ------------------

/** @brief Call the GPU kernel calculating loss function on GPU.
 *  @param p_model Pointer to canonical decomposition model pre-allocated on GPU.
 *  @param p_train_data Pointer to train data array pre-allocated on GPU.
 *  @param p_result Pointer to result array on GPU.
 *  @param shared_mem_size Size (in bytes) of the block-wise shared memory.
 *  @param stream_ptr Pointer to the CUDA calculation stream in form of an unsigned integer pointer.
 *  @param n_thread Number of CUDA threads for parallel execution.
 *  @note This function is asynchronous. It simply push the CUDA kernel to the stream.
 */
void call_loss_function_kernel(const candy::Model * p_model, const array::Parcel * p_train_data, double * p_result,
                               std::uint64_t shared_mem_size, std::uintptr_t stream_ptr, std::uint64_t n_thread);

/** @brief Call the GPU kernel calculating the gradient on GPU.
 *  @param p_model Pointer to canonical decomposition model pre-allocated on GPU.
 *  @param p_train_data Pointer to train data array pre-allocated on GPU.
 *  @param p_gradient Pointer to result array on GPU.
 *  @param shared_mem_size Size (in bytes) of the block-wise shared memory.
 *  @param stream_ptr Pointer to the CUDA calculation stream in form of an unsigned integer pointer.
 *  @param n_thread Number of CUDA threads for parallel execution.
 *  @note This function is asynchronous. It simply push the CUDA kernel to the stream.
 */
void call_model_gradient_kernel(const candy::Model * p_model, const array::Parcel * p_train_data, double * p_gradient,
                                std::uint64_t shared_mem_size, std::uintptr_t stream_ptr, std::uint64_t n_thread);

// Loss function
// -------------

/** @brief Calculate loss function.
 *  @details Calculate loss function of a model over a dataset with CPU parallelism.
 */
MERLIN_EXPORTS double calc_loss_function_cpu(const candy::Model & model, const array::Array & train_data);

/** @brief Calculate loss function.
 *  @details Calculate loss function of a model over a dataset with GPU parallelism.
 *  @param model Canonical decomposition model.
 *  @param train_data Reference data to calculate the loss function.
 *  @param result A vector of floating points, of size n_thread for storing partial loss function calculated by each
 *  CUDA thread. The value of the actual loss function is the sum of all entries of the vector.
 *  @param stream CUDA calculation stream.
 *  @param n_thread Number of CUDA threads to calculate the loss function.
 */
void calc_loss_function_gpu(const candy::Model & model, const array::Parcel & train_data, floatvec & result,
                            const cuda::Stream & stream = cuda::Stream(),
                            std::uint64_t n_thread = Environment::default_block_size);

// Model gradient
// --------------

/** @brief Convert contiguous index to ndim index with a dimension fixed.*/
__cuhostdev__ intvec contiguous_to_ndim_idx_1(std::uint64_t index, const intvec & shape, std::uint64_t skip_dim,
                                              std::uint64_t * data_ptr = nullptr);

/** @brief Calculate gradient of a model.
 *  @details Calculate gradient of loss function of a model over a dataset with CPU parallelism.
 *  @param model Canonical decomposition model.
 *  @param train_data Target data to fit the model.
 */
MERLIN_EXPORTS Vector<double> calc_gradient_vector_cpu(const candy::Model & model, const array::Array & train_data);

/** @brief Calculate gradient of a model.
 *  @details Calculate gradient of loss function of a model over a dataset with GPU parallelism.
 */
void calc_gradient_vector_gpu(const candy::Model & model, const array::Parcel & train_data, floatvec & result,
                              const cuda::Stream & stream = cuda::Stream(),
                              std::uint64_t n_thread = Environment::default_block_size);

}  // namespace merlin::candy

#endif  // MERLIN_GRADIENT_DESCENT_HPP_
