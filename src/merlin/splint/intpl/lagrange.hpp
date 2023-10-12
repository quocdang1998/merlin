// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTPL_LAGRANGE_HPP_
#define MERLIN_SPLINT_INTPL_LAGRANGE_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::splint::intpl {

/** @brief Construct interpolation coefficients by Lagrange method on CPU.
 *  @param coeff Pointer to the first element in the coefficient array.
 *  @param grid_nodes Pointer to the grid node array of the current dimension.
 *  @param shape Number of nodes on the current dimension.
 *  @param element_size Size of each sub-array element.
 *  @param thread_idx Index of the thread in group.
 *  @param n_threads Number of threads performing the action.
 */
MERLIN_EXPORTS void construction_lagrange_cpu(double * coeff, const double * grid_nodes, std::uint64_t shape,
                                              std::uint64_t element_size, std::uint64_t thread_idx,
                                              std::uint64_t n_threads) noexcept;

/** @brief Interpolate recursively on each dimension.
 *  @param coeff C-contiguous array of coefficients.
 *  @param num_coeff Size of coefficient array.
 *  @param c_index_coeff C-contiguous index of the current coefficient.
 *  @param ndim_index_coeff Multi-dimensional index of the current coefficient.
 *  @param cache_array Pointer to cache memory.
 *  @param i_dim Index of the current dimension.
 *  @param ndim Number of dimension.
 *  @param grid_nodes Pointer to the grid node array of the current dimension.
 *  @param last_dim_nodes Pointer to the grid node array of the last dimension.
 *  @param point Coordinates of the point.
 */
MERLIN_EXPORTS void eval_lagrange_cpu(const double * coeff, const std::uint64_t & num_coeff,
                                      const std::uint64_t & c_index_coeff, const std::uint64_t * ndim_index_coeff,
                                      double * cache_array, const double * point, const std::int64_t & i_dim,
                                      const std::uint64_t * grid_shape, double * const * grid_vectors,
                                      const std::uint64_t & ndim);

}  // namespace merlin::splint::intpl

#endif  // MERLIN_SPLINT_INTPL_LAGRANGE_HPP_
