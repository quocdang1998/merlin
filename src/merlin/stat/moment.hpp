// Copyright 2023 quocdang1998
#ifndef MERLIN_STAT_MOMENT_HPP_
#define MERLIN_STAT_MOMENT_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t

#include "merlin/array/nddata.hpp"    // merlin::array::Array, merlin::array::Parcel, merlin::array::Stock
#include "merlin/cuda_interface.hpp"  // __cudevice__
#include "merlin/cuda/stream.hpp"     // merlin::cuda::Stream
#include "merlin/exports.hpp"         // MERLIN_EXPORTS
#include "merlin/vector.hpp"          // merlin::floatvec

namespace merlin::stat {

// Powered Mean
// ------------

/** @brief Calculate powered mean on a CPU array.
 *  @details Calculate @f$ \mathbb{E}[X^k] @f$, @f$ k @f$ is the order.
 *  @param order Order to calculate.
 *  @param data Input data.
 *  @param n_threads Number of threads.
 */
MERLIN_EXPORTS floatvec powered_mean(std::uint64_t order, const array::Array & data, std::uint64_t n_threads = 1);

/** @brief Calculate powered mean on a GPU array.
 *  @details Calculate @f$ \mathbb{E}[X^k] @f$, @f$ k @f$ is the order.
 *  @param order Order to calculate.
 *  @param data Input data.
 *  @param n_threads Number of threads.
 *  @param stream CUDA stream.
 */
MERLIN_EXPORTS floatvec powered_mean(std::uint64_t order, const array::Parcel & data, std::uint64_t n_threads = 1,
                                     const cuda::Stream & stream = cuda::Stream());

// Mean
// ----

/** @brief Calculate mean on all elements of an array.*/
inline double mean(const array::Array & data, std::uint64_t n_threads = 1) {
    return powered_mean(1, data, n_threads)[0];
}

/** @brief Calculate mean on all elements of an array.*/
inline double mean(const array::Parcel & data, std::uint64_t n_threads = 1,
                   const cuda::Stream & stream = cuda::Stream()) {
    floatvec mean(powered_mean(1, data, n_threads, stream));
    stream.synchronize();
    return mean[0];
}

// Mean and variance
// -----------------

/** @brief Calculate mean and variance on all elements of an array.*/
inline std::array<double, 2> mean_variance(const array::Array & data, std::uint64_t n_threads = 1) {
    floatvec mean_var = powered_mean(2, data, n_threads);
    return std::array<double, 2>({mean_var[0], mean_var[1] - mean_var[0] * mean_var[0]});
}

/** @brief Calculate mean and variance on all elements of an array.*/
inline std::array<double, 2> mean_variance(const array::Parcel & data, std::uint64_t n_threads = 1,
                                           const cuda::Stream & stream = cuda::Stream()) {
    floatvec mean_var = powered_mean(2, data, n_threads, stream);
    stream.synchronize();
    return std::array<double, 2>({mean_var[0], mean_var[1] - mean_var[0] * mean_var[0]});
}

}  // namespace merlin

#endif  // MERLIN_STAT_MOMENT_HPP_
