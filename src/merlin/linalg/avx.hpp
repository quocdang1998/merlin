// Copyright 2024 quocdang1998
#ifndef MERLIN_LINALG_KERNEL_AVX_HPP_
#define MERLIN_LINALG_KERNEL_AVX_HPP_

#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint64_t

namespace merlin::linalg {

// Vector size
// -----------

/** @brief Vector size.*/
inline constexpr std::uint64_t pack_size = __MERLIN_VECTOR_SIZE__;

/** @brief AVX alignment.*/
inline constexpr std::size_t vector_size = sizeof(double) * pack_size;

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_KERNEL_AVX_HPP_
