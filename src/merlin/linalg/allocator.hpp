// Copyright 2024 quocdang1998
#ifndef MERLIN_LINALG_ALLOCATOR_HPP_
#define MERLIN_LINALG_ALLOCATOR_HPP_

#include <cstddef>  // std::size_t

namespace merlin::linalg {

// Aligned allocator
// -----------------

/** @brief Wrapper for aligned alloc.*/
double * aligned_alloc(std::size_t alignment, std::size_t size);

/** @brief Wrapper for aligned free.*/
void aligned_free(double * ptr);

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_ALLOCATOR_HPP_
