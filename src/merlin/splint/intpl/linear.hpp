// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTPL_LINEAR_HPP_
#define MERLIN_SPLINT_INTPL_LINEAR_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/cuda_interface.hpp"  // __cuhostdev__

namespace merlin::splint::intpl {

/** @brief Construct interpolation coefficients by linear interpolation method.
 *  @param coeff Pointer to the first element in the coefficient array.
 *  @param grid_nodes Pointer to the grid node array of the current dimension.
 *  @param shape Number of nodes on the current dimension.
 *  @param element_size Size of each sub-array element.
 *  @param thread_idx Index of the thread in group.
 *  @param n_threads Number of threads performing the action.
 */
__cuhostdev__ void construct_linear(double * coeff, const double * grid_nodes, const std::uint64_t & shape,
                                    const std::uint64_t & element_size, const std::uint64_t & thread_idx,
                                    const std::uint64_t & n_threads) noexcept;

/** @brief Evaluate interpolation at an unit step by linear interpolation method.
 *  @param grid_nodes Array of nodes to interpolate.
 *  @param grid_shape Number of nodes.
 *  @param point Coordinate of the point.
 *  @param coeff_index Index of the coefficient.
 *  @param coeff Value of interpolation coefficient.
 *  @param result Variable to which the result of the interpolation is added to.
 */
__cuhostdev__ void evaluate_linear(const double * grid_nodes, const std::uint64_t & grid_shape, const double & point,
                                   const std::uint64_t & coeff_index, const double & coeff, double & result) noexcept;

}  // namespace merlin::splint::intpl

#endif  // MERLIN_SPLINT_INTPL_LINEAR_HPP_
