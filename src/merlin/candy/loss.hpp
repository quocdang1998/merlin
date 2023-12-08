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

__cuhostdev__ inline void rmse_updater(double & current_error, const double & predicted, const double & reference) {
    double error = (predicted - reference) / reference;
    current_error += error * error;
}

__cuhostdev__ double rmse_averager(const double * thread_result_data, const std::uint64_t * non_zero_element_data,
                                   std::uint64_t n_threads);

/** @brief Calculate relative mean square error with CPU parallelism.
 *  @details Calculate:
 *  @f[ \frac{1}{N} \left[ \sum_{\text{data}} \left( \frac{x_{\text{model}} - x_{\text{data}}}{x_{\text{data}}}
 *  \right)^2 \right] @f]
 *
 * in which @f$ N @f$ is the number of non-zero elements in the data, @f$ x_{\text{model}} @f$ is the value predicted
 * by the model, and @f$ x_{\text{data}} @f$ is the value from the train data.
 */
MERLIN_EXPORTS double rmse_cpu(const candy::Model * p_model, const array::Array * p_train_data,
                               std::uint64_t n_threads = 1) noexcept;

/** @brief Calculate relative mean square error with GPU parallelism.
 *  @param p_model Pointer to Candecomp model on GPU.
 *  @param p_train_data Pointer to train data on GPU.
 *  @param ndim Number of dimension of model and train data.
 *  @param n_threads Number of parallel threads.
 *  @param share_mem Share memory, should be at least ``model.share_mem_size() + data.share_mem_size()``.
 */
double rmse_gpu(const candy::Model * p_model, const array::Parcel * p_train_data, std::uint64_t ndim,
                std::uint64_t share_mem, std::uint64_t n_threads = 1) noexcept;

// Relative max absolute error
// ---------------------------

__cuhostdev__ inline void rmae_updater(double & current_error, const double & predicted, const double & reference) {
    double error = std::abs(predicted - reference) / reference;
    current_error = (error > current_error) ? error : current_error;
}

__cuhostdev__ double rmae_averager(const double * thread_result_data, const std::uint64_t * non_zero_element_data,
                                   std::uint64_t n_threads);

/** @brief Calculate relative max error with CPU parallelism.*/
MERLIN_EXPORTS double rmae_cpu(const candy::Model * p_model, const array::Array * p_train_data,
                               std::uint64_t n_threads = 1) noexcept;

/** @brief Calculate relative max absolute error with GPU parallelism.
 *  @param p_model Pointer to Candecomp model on GPU.
 *  @param p_train_data Pointer to train data on GPU.
 *  @param ndim Number of dimension of model and train data.
 *  @param n_threads Number of parallel threads.
 *  @param share_mem Share memory, should be at least ``model.share_mem_size() + data.share_mem_size()``.
 */
double rmae_gpu(const candy::Model * p_model, const array::Parcel * p_train_data, std::uint64_t ndim,
                std::uint64_t share_mem, std::uint64_t n_threads = 1) noexcept;

}  // namespace merlin::candy

#endif  // MERLIN_CANDY_LOSS_HPP_
