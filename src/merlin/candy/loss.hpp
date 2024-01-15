// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_LOSS_HPP_
#define MERLIN_CANDY_LOSS_HPP_

#include <cmath>        // std::abs
#include <type_traits>  // std::add_pointer

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Parcel
#include "merlin/candy/declaration.hpp"  // merlin::candy::Model
#include "merlin/cuda/stream.hpp"        // merlin::cuda::Stream
#include "merlin/cuda_interface.hpp"     // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"            // MERLIN_EXPORTS
#include "merlin/vector.hpp"             // merlin::floatvec, merlin::intvec

namespace merlin::candy {

// Relative mean square error
// --------------------------

/** @brief Calculate relative mean square error with CPU parallelism.
 *  @details Calculate:
 *  @f[ \sqrt{ \frac{1}{N} \left[ \sum_{\text{data}} \left( \frac{x_{\text{model}} - x_{\text{data}}}{x_{\text{data}}}
 *  \right)^2 \right] } @f]
 *
 *  in which @f$ N @f$ is the number of non-zero elements in the data, @f$ x_{\text{model}} @f$ is the value predicted
 *  by the model, and @f$ x_{\text{data}} @f$ is the value from the train data.
 *  @warning This function will lock the mutex.
 *  @param p_model Pointer to CP model.
 *  @param p_data Pointer to train data.
 *  @param buffer Pointer to buffer memory for calculation, should be at least ``std::uint64_t[n_threads*ndim]``.
 *  @param n_threads Number of threads to calculate.
 */
MERLIN_EXPORTS double rmse_cpu(const candy::Model * p_model, const array::Array * p_data, std::uint64_t * buffer,
                               std::uint64_t n_threads = 1) noexcept;

#ifdef __NVCC__

/** @brief CUDA function calculating relative mean square error with GPU parallelism.
 *  @details Calculate:
 *  @f[ \sqrt{ \frac{1}{N} \left[ \sum_{\text{data}} \left( \frac{x_{\text{model}} - x_{\text{data}}}{x_{\text{data}}}
 *  \right)^2 \right] } @f]
 *
 *  in which @f$ N @f$ is the number of non-zero elements in the data, @f$ x_{\text{model}} @f$ is the value predicted
 *  by the model, and @f$ x_{\text{data}} @f$ is the value from the train data.
 *  @param p_model Pointer to CP model.
 *  @param p_data Pointer to train data.
 *  @param buffer Buffer memory for calculation, should be at least ``std::uint64_t[ndim*block_size]``
 *  @param p_result Pointer to the result.
 *  @param thread_idx Index of the current thread in the thread block.
 *  @param block_size Number of thread in a block.
 */
__cudevice__ void rmse_gpu(const candy::Model * p_model, const array::Parcel * p_data, std::uint64_t * buffer,
                           double * p_result, std::uint64_t thread_idx, std::uint64_t block_size) noexcept;

#endif  // __NVCC__

// Relative max absolute error
// ---------------------------

/** @brief Calculate relative max absolute error with CPU parallelism.
 *  @details Calculate:
 *  @f[ \max_{\text{data}} \left[ \frac{\left| x_{\text{model}} - x_{\text{data}} \right|}{x_{\text{data}}} \right] @f]
 *
 *  in which @f$ N @f$ is the number of non-zero elements in the data, @f$ x_{\text{model}} @f$ is the value predicted
 *  by the model, and @f$ x_{\text{data}} @f$ is the value from the train data.
 *  @warning This function will lock the mutex.
 *  @param p_model Pointer to CP model.
 *  @param p_data Pointer to train data.
 *  @param buffer Pointer to buffer memory for calculation, should be at least ``std::uint64_t[n_threads*ndim]``.
 *  @param n_threads Number of threads to calculate.
 */
MERLIN_EXPORTS double rmae_cpu(const candy::Model * p_model, const array::Array * p_data, std::uint64_t * buffer,
                               std::uint64_t n_threads = 1) noexcept;

#ifdef __NVCC__

/** @brief CUDA function calculating relative max absolute error with GPU parallelism.
 *  @details Calculate:
 *  @f[ \max_{\text{data}} \left[ \frac{\left| x_{\text{model}} - x_{\text{data}} \right|}{x_{\text{data}}} \right] @f]
 *
 *  in which @f$ N @f$ is the number of non-zero elements in the data, @f$ x_{\text{model}} @f$ is the value predicted
 *  by the model, and @f$ x_{\text{data}} @f$ is the value from the train data.
 *  @param p_model Pointer to CP model.
 *  @param p_data Pointer to train data.
 *  @param buffer Buffer memory for calculation, should be at least ``std::uint64_t[ndim*block_size]``
 *  @param p_result Pointer to the result.
 *  @param thread_idx Index of the current thread in the thread block.
 *  @param block_size Number of thread in a block.
 */
__cudevice__ void rmae_gpu(const candy::Model * p_model, const array::Parcel * p_data, std::uint64_t * buffer,
                           double * p_result, std::uint64_t thread_idx, std::uint64_t block_size) noexcept;

#endif  // __NVCC__

}  // namespace merlin::candy

#endif  // MERLIN_CANDY_LOSS_HPP_
