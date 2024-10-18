// Copyright 2024 quocdang1998
#ifndef MERLIN_SIMD_ALIGNED_ALLOCATOR_HPP_
#define MERLIN_SIMD_ALIGNED_ALLOCATOR_HPP_

#include <cstddef>  // std::size_t

namespace merlin::simd {

// Functions for allocating and freeing aligned memory
// ---------------------------------------------------

/** @brief Wrapper for aligned alloc.
 *  @param alignment Alignment constraint. The returned pointer is guaranteed to be divisible by this number.
 *  @param size Number of double-precision elements.
 */
double * aligned_alloc(std::size_t alignment, std::size_t size);

/** @brief Wrapper for aligned free.*/
void aligned_free(double * ptr);

}  // namespace merlin::simd

#endif  // MERLIN_SIMD_ALIGNED_ALLOCATOR_HPP_
