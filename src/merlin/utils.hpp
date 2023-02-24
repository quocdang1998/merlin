// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include <string>  // std::string

#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_HOSTDEV_EXPORTS
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// System
// ------

/** @brief Get process ID in form of a string.*/
std::string get_current_process_id(void);

/** @brief Get current time in form of a string.*/
std::string get_time(void);

// Multi-dimensional Index
// -----------------------

/** @brief Inner product of 2 index vectors.
 *  @details Return convolution product / scalar product of 2 vectors.
 *  @param v1 First vector.
 *  @param v2 Second vector.
 */
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS std::uint64_t inner_prod(const intvec & v1, const intvec & v2);

/** @brief Convert n-dimensional index to C-contiguous index.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return C-contiguous index as an ``std::uint64_t``.
 */
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS std::uint64_t ndim_to_contiguous_idx(const intvec & index, const intvec & shape);

/** @brief Convert C-contiguous index to n-dimensional index.
 *  @param index C-contiguous index.
 *  @param shape Shape vector.
 *  @return merlin::intvec of n-dimensional index.
 */
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS intvec contiguous_to_ndim_idx(std::uint64_t index, const intvec & shape);

// Sparse Grid
// -----------

/** @brief Get size of a sub-grid given its level vector.*/
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS std::uint64_t calc_subgrid_size(const intvec & level_vector) noexcept;

/** @brief Get shape of Cartesian subgrid corresponding to a level vector.*/
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS intvec get_level_shape(const intvec & level_vector);











/** @brief Get size of a 1D grid given its max level.*/
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS constexpr std::uint64_t get_size_from_level(std::uint64_t level) noexcept {
    return (level == 0) ? 1 : ((1 << level) + 1);
}

/** @brief Index of nodes belonging to a level of a 1D grid.
 *  @param level Level to get index.
 *  @param size Size of 1D grid level.
 */
// __cuhostdev__ intvec hiearchical_index(std::uint64_t level, std::uint64_t size);  // x1

}  // namespace merlin

#endif  // MERLIN_UTILS_HPP_
