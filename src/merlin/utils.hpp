// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// Multi-dimensional Index
// -----------------------

/** @brief Inner product of 2 index vectors.
 *  @details Return convolution product / scalar product of 2 vectors.
 *  @param v1 First vector.
 *  @param v2 Second vector.
 */
__cuhostdev__ std::uint64_t inner_prod(const intvec & v1, const intvec & v2);

/** @brief Convert n-dimensional index to C-contiguous index.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return C-contiguous index as an ``std::uint64_t``.
 */
__cuhostdev__ std::uint64_t ndim_to_contiguous_idx(const intvec & index, const intvec & shape);

/** @brief Convert C-contiguous index to n-dimensional index.
 *  @param index C-contiguous index.
 *  @param shape Shape vector.
 *  @return merlin::intvec of n-dimensional index.
 */
__cuhostdev__ intvec contiguous_to_ndim_idx(std::uint64_t index, const intvec & shape);

}  // namespace merlin

#endif  // MERLIN_UTILS_HPP_
