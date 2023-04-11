// Copyright 2023 quocdang1998
#ifndef MERLIN_STATISTICS_MOMENT_HPP_
#define MERLIN_STATISTICS_MOMENT_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t

#include "merlin/array/nddata.hpp"    // merlin::array::Array, merlin::array::Parcel
#include "merlin/cuda_interface.hpp"  // __cudevice__
#include "merlin/env.hpp"             // merlin::Environment
#include "merlin/exports.hpp"         // MERLIN_EXPORTS
#include "merlin/vector.hpp"          // merlin::intvec

namespace merlin::statistics {

/** @brief Calculate generalized mean of elements in an array.
 *  @details Calculate @f$ \mathbb{E}[X^k] @f$, @f$ k @f$ is the order.
 *  @param data Data to calculate the mean.
 *  @param nthreads Number of CPU threads to perform the calculation.
 *  @returns Array of @f$ \mathbb{E}[X^i] @f$ for @f$ 1 \le i \le k @f$.
 */
template <std::uint64_t order>
std::array<double, order> powered_mean(const array::Array & data,
                                       std::uint64_t nthreads = Environment::default_block_size);

/** @brief Calculate moment from generalized mean vector.
 *  @details Calculate @f$ \mathbb{E}[(X-\mathbb{E}[X])^k] @f$, @f$ k @f$ is the order.
 *  @param powered_means Array of powered means (result of function statistics::powered_mean).
 */
template <std::uint64_t order>
double moment_cpu(const std::array<double, order> & powered_means);

/** @brief Calculate mean on all elements of the array.
 *  @param data Array of data to calculate the mean.
 *  @param nthreads Number of CPU threads to perform the calculation.
 */
MERLIN_EXPORTS double mean_cpu(const array::Array & data, std::uint64_t nthreads = Environment::default_block_size);

/** @brief Calculate variance on all elements of the array.
 *  @param data Array of data to calculate the mean.
 *  @param nthreads Number of CPU threads to perform the calculation.
 */
MERLIN_EXPORTS double variance_cpu(const array::Array & data, std::uint64_t nthreads = Environment::default_block_size);

/** @brief Calculate max element of the array.
 *  @param data Array of data to calculate the mean.
 *  @param nthreads Number of CPU threads to perform the calculation.
 */
MERLIN_EXPORTS double max_cpu(const array::Array & data, std::uint64_t nthreads = Environment::default_block_size);

/** @brief Calculate mean for a given set of dimensions.
 *  @param data Array of data to calculate the mean.
 *  @param dims Dimension on which the mean is calculated.
 *  @param nthreads Number of CPU threads to perform the calculation.
 */
MERLIN_EXPORTS array::Array mean_cpu(const array::Array & data, const intvec & dims,
                                     std::uint64_t nthreads = Environment::default_block_size);

#ifdef __NVCC__

/** @brief Calculate mean on all elements of the array.
 *  @param data Array of data to calculate the mean.
 *  @param buffer Buffer memory for calculation.
 */
__cudevice__ void mean_gpu(const array::Parcel & data, double * buffer);

#endif  // __NVCC__

}  // namespace merlin::statistics

#include "merlin/statistics/moment.tpp"

#endif  // MERLIN_STATISTICS_MOMENT_HPP_
