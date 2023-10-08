// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTPL_LINEAR_HPP_
#define MERLIN_SPLINT_INTPL_LINEAR_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::splint::intpl {

/** @brief Construct interpolation coefficients by linear interpolation method on CPU.
 *  @param coeff Pointer to the first element in the coefficient array.
 *  @param grid_nodes Pointer to the grid node array of the current dimension.
 *  @param shape Number of nodes on the current dimension.
 *  @param element_size Size of each sub-array element.
 *  @param n_threads Number of threads performing the action.
 */
MERLIN_EXPORTS void construction_linear_cpu(double * coeff, double * grid_nodes, std::uint64_t shape,
                                            std::uint64_t element_size, std::uint64_t n_threads) noexcept;

}  // namespace merlin::splint::intpl

#endif  // MERLIN_SPLINT_INTPL_LINEAR_HPP_
