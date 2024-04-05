// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_TOOLS_HPP_
#define MERLIN_SPLINT_TOOLS_HPP_

#include <future>       // std::shared_future
#include <type_traits>  // std::add_pointer

#include "merlin/cuda_interface.hpp"       // __cuhostdev__
#include "merlin/cuda/declaration.hpp"     // merlin::cuda::Stream
#include "merlin/exports.hpp"              // MERLIN_EXPORTS
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/vector.hpp"               // merlin::Vector

namespace merlin::splint {

// Construct Coefficients
// ----------------------

/** @brief Type of construction methods.*/
using ConstructionMethod = std::add_pointer<void(double *, const double *, const std::uint64_t &, const std::uint64_t &,
                                                 const std::uint64_t &, const std::uint64_t &)>::type;

/** @brief Construct interpolation coefficients with CPU parallelism.
 *  @param current_job Pointer to previous asynchronous job.
 *  @param coeff C-contiguous array of coefficients (value are pre-copied to this array).
 *  @param p_grid Pointer to Cartesian grid to interpolate.
 *  @param p_method Pointer to interpolation method to use on each dimension.
 *  @param n_threads Number of threads to calculate.
 */
void construct_coeff_cpu(std::shared_future<void> current_job, double * coeff, const grid::CartesianGrid * p_grid,
                         const Vector<unsigned int> * p_method, std::uint64_t n_threads) noexcept;

/** @brief Construct interpolation coefficients with GPU parallelism.
 *  @param coeff C-contiguous array of coefficients on GPU (value are pre-copied to this array).
 *  @param p_grid Pointer to Cartesian grid to interpolate (pre-copied to GPU).
 *  @param p_method Pointer to interpolation method to use on each dimension.
 *  @param n_threads Number of threads to calculate.
 *  @param shared_mem_size Size of share memory (at least ``p_grid`` and ``p_method``).
 *  @param stream_ptr Pointer to the CUDA stream performing this calculation.
 */
void construct_coeff_gpu(double * coeff, const grid::CartesianGrid * p_grid, const Vector<unsigned int> * p_method,
                         std::uint64_t n_threads, std::uint64_t shared_mem_size,
                         const cuda::Stream * stream_ptr) noexcept;

// Evaluate Interpolation
// ----------------------

/** @brief Type of evaluation methods.*/
using EvaluationMethod = std::add_pointer<void(const double *, const std::uint64_t &, const double &,
                                               const std::uint64_t &, const double &, double &)>::type;

/** @brief Interpolate recursively on each dimension.
 *  @param coeff C-contiguous array of coefficients.
 *  @param num_coeff Size of coefficient array.
 *  @param c_index_coeff C-contiguous index of the current coefficient.
 *  @param ndim_index_coeff Multi-dimensional index of the current coefficient.
 *  @param cache_array Pointer to cache memory.
 *  @param point Coordinates of the point.
 *  @param i_dim Index of the current dimension.
 *  @param grid_shape Grid shape array.
 *  @param grid_vectors Array of array of nodes in the grid.
 *  @param method Method vector to interpolate.
 *  @param ndim Number of dimension.
 */
__cuhostdev__ void recursive_interpolate(const double * coeff, const std::uint64_t & num_coeff,
                                         const std::uint64_t & c_index_coeff, const std::uint64_t * ndim_index_coeff,
                                         double * cache_array, const double * point, const std::int64_t & i_dim,
                                         const std::uint64_t * grid_shape, double * const * grid_vectors,
                                         const Vector<unsigned int> * p_method, const std::uint64_t & ndim) noexcept;

/** @brief Evaluate interpolation with CPU parallelism.
 *  @param current_job Pointer to previous asynchronous job.
 *  @param coeff C-contiguous array of coefficients.
 *  @param p_grid Pointer to Cartesian grid to interpolate.
 *  @param p_method Pointer to interpolation method to use on each dimension.
 *  @param points Pointer to the first coordinate of the first point. Coordinates of the same point are placed
 *  side-by-side in the array.
 *  @param n_points Number of points to interpolate.
 *  @param result Pointer to the array storing the result.
 *  @param n_threads Number of threads to perform the interpolation.
 */
void eval_intpl_cpu(std::shared_future<void> current_job, const double * coeff, const grid::CartesianGrid * p_grid,
                    const Vector<unsigned int> * p_method, const double * points, std::uint64_t n_points,
                    double * result, std::uint64_t n_threads) noexcept;

/** @brief Evaluate interpolation with GPU parallelism.
 *  @param coeff C-contiguous array of coefficients on GPU (value are pre-copied to this array).
 *  @param p_grid Pointer to Cartesian grid to interpolate on GPU.
 *  @param p_method Pointer to vector of interpolation method to use on each dimension on GPU.
 *  @param points Pointer to the first coordinate of the first point on GPU. Coordinates of the same point are placed
 *  side-by-side in the array.
 *  @param n_points Number of points to interpolate.
 *  @param result Pointer to the array storing the result on GPU.
 *  @param n_threads Number of threads to calculate.
 *  @param ndim Number of dimension of the grid and data.
 *  @param shared_mem_size Size of share memory (at least ``p_grid``, ``p_method``).
 *  @param stream_ptr Pointer to the CUDA stream performing this calculation.
 */
void eval_intpl_gpu(double * coeff, const grid::CartesianGrid * p_grid, const Vector<unsigned int> * p_method,
                    double * points, std::uint64_t n_points, double * result, std::uint64_t n_threads,
                    std::uint64_t ndim, std::uint64_t shared_mem_size, const cuda::Stream * stream_ptr) noexcept;

}  // namespace merlin::splint

#endif  // MERLIN_SPLINT_TOOLS_HPP_
