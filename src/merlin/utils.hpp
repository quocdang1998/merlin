// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include <array>   // std::array
#include <string>  // std::string

#include "merlin/config.hpp"    // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"   // MERLIN_EXPORTS
#include "merlin/platform.hpp"  // __MERLIN_LINUX__
#include "merlin/vector.hpp"    // merlin::UIntVec

#if defined(__MERLIN_LINUX__)
    #include <cmath>  // std::isfinite, std::isnormal
#elif defined(__MERLIN_WINDOWS__)
    #include <math.h>  // isfinite, isnormal
#endif                 // __MERLIN_LINUX__

namespace merlin {

// System
// ------

/** @brief Get process ID in form of a string.*/
MERLIN_EXPORTS std::string get_current_process_id(void);

/** @brief Get current time in form of a string.
 *  @details Return datetime in form of ``{year}-{month}-{day}_{hour}:{minute}:{second}``.
 */
MERLIN_EXPORTS std::string get_time(void);

// Math check normal
// -----------------

// Check if a value is normal
__cuhostdev__ inline bool is_normal(double value) noexcept {
#if defined(__MERLIN_LINUX__) && !defined(__CUDA_ARCH__)
    return std::isnormal(value);
#else
    return isfinite(value) && (value != 0.0);
#endif  // __MERLIN_LINUX__ && !__CUDA_ARCH__
}

// Check if a value is finite
__cuhostdev__ inline bool is_finite(double value) noexcept {
#if defined(__MERLIN_LINUX__) && !defined(__CUDA_ARCH__)
    return std::isfinite(value);
#else
    return isfinite(value);
#endif  // __MERLIN_LINUX__ && !__CUDA_ARCH__
}

// CUDA kernel
// -----------

/** @brief Calculate number of block in a grid give the size of each block and the size of the loop.*/
__cuhostdev__ constexpr std::uint64_t get_block_count(std::uint64_t block_size, std::uint64_t loop_size) {
    return (loop_size + block_size - 1) / block_size;
}

#ifdef __NVCC__

/** @brief Thread index in block.
 *  @details Get the three-dimensional flattened index of the current thread in the current block.
 */
__cudevice__ inline std::uint64_t flatten_thread_index(void) {
    std::uint64_t result = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    return result;
}

/** @brief Size of block.
 *  @details Get the number of threads in the current block.
 */
__cudevice__ inline std::uint64_t size_of_block(void) {
    std::uint64_t result = blockDim.x * blockDim.y * blockDim.z;
    return result;
}

/** @brief Block index in grid.
 *  @details Get the three-dimensional flattened index of the current block in the current grid.
 */
__cudevice__ inline std::uint64_t flatten_block_index(void) {
    std::uint64_t result = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    return result;
}

/** @brief Thread index in grid.
 *  @details Get the index of the current thread in the current grid.
 */
__cudevice__ inline std::uint64_t flatten_kernel_index(void) {
    std::uint64_t index_in_block = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    std::uint64_t index_of_block = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    std::uint64_t size_of_one_block = blockDim.x * blockDim.y * blockDim.z;
    return index_of_block * size_of_one_block + index_in_block;
}

#endif  // __NVCC__

// Bit flip
// --------

constexpr std::uint64_t flip_endianess(std::uint64_t value) {
    return ((value & 0xFFULL) << 56) | ((value & 0xFF00ULL) << 40) | ((value & 0xFF0000ULL) << 24) |
           ((value & 0xFF000000ULL) << 8) | ((value & 0xFF00000000ULL) >> 8) | ((value & 0xFF0000000000ULL) >> 24) |
           ((value & 0xFF000000000000ULL) >> 40) | ((value & 0xFF00000000000000ULL) >> 56);
}

inline double flip_endianess(double value) {
    std::uint64_t * p_int_value = reinterpret_cast<std::uint64_t *>(&value);
    std::uint64_t flipped_int = flip_endianess(*p_int_value);
    double * flipped_real = reinterpret_cast<double *>(&flipped_int);
    return *flipped_real;
}

inline void flip_range(std::uint64_t * range, std::uint64_t length) {
    for (std::uint64_t i = 0; i < length; i++) {
        range[i] = flip_endianess(range[i]);
    }
}

// Multi-dimensional Index
// -----------------------

/** @brief Product of entries of a vector.
 *  @details Return product of all elements of the vector.
 *  @param v Vector.
 *  @param size Number of element in the vector.
 *  @note This function returns ``1`` if ``v`` has zero size.
 */
__cuhostdev__ std::uint64_t prod_elements(const std::uint64_t * v, std::uint64_t size);

/** @brief Inner product of 2 index vectors.
 *  @details Return convolution product / scalar product of 2 vectors.
 *  @param v1 First vector.
 *  @param v2 Second vector.
 *  @param size Size of 2 vectors.
 */
__cuhostdev__ std::uint64_t inner_prod(const std::uint64_t * v1, const std::uint64_t * v2, std::uint64_t size);

/** @brief Convert n-dimensional index to C-contiguous index.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @param ndim Number of dimension.
 *  @return C-contiguous index as an ``std::uint64_t``.
 */
__cuhostdev__ std::uint64_t ndim_to_contiguous_idx(const std::uint64_t * index, const std::uint64_t * shape,
                                                   std::uint64_t ndim);

/** @brief Convert C-contiguous index to n-dimensional index and save data to a pre-allocated memory.
 *  @param index C-contiguous index.
 *  @param shape Shape vector.
 *  @param ndim Number of dimension.
 *  @param data_ptr Pointer to result data.
 */
__cuhostdev__ void contiguous_to_ndim_idx(std::uint64_t index, const std::uint64_t * shape, std::uint64_t ndim,
                                          std::uint64_t * data_ptr);

/** @brief Increase an n-dimensional index by one unit.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @param ndim Number of dimension.
 *  @return Lowest changed dimension.
 */
__cuhostdev__ std::int64_t increment_index(std::uint64_t * index, const std::uint64_t * shape, std::uint64_t ndim);

/** @brief Decrease an n-dimensional index by one unit.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @param ndim Number of dimension.
 *  @return Lowest changed dimension.
 */
__cuhostdev__ std::int64_t decrement_index(std::uint64_t * index, const std::uint64_t * shape, std::uint64_t ndim);

// List Division
// -------------

/** @brief Get a list of pointers divided from an original array.
 *  @param original Pointer to the first element of the original array.
 *  @param divider_length Size of each subsequence.
 *  @param num_seq Number of sequences.
 *  @param data_ptr Pointer to result data. If the value is ``nullptr``, new instance is allocated.
 */
__cuhostdev__ void ptr_to_subsequence(double * original, const std::uint64_t * divider_length, std::uint64_t num_seq,
                                      double ** data_ptr);

/** @brief Get index of sequence and index in that sequence of an index in original array.
 *  @param index_full_array Index in original array.
 *  @param divider_length Size of each subsequence dividing the original array.
 *  @param num_seq Number of subsequences.
 *  @returns Index of the list, and index of the element in the list.
 *  @note If the index in full array overpass the last element in the sequence, the function will return the number of
 *  dimension, and the offset with respect to the last element in the sequence.
 */
__cuhostdev__ std::array<std::uint64_t, 2> index_in_subsequence(std::uint64_t index_full_array,
                                                                const std::uint64_t * divider_length,
                                                                std::uint64_t num_seq) noexcept;

// Triangular Index
// ----------------

/** @brief Get 2-dimensional triangular index from flatten index.
 *  @details Decompose the index @f$ i @f$ into:
 *  @f[ i = T_k + r @f]
 *  in which @f$ T_k @f$ (@f$ k \ge 0 @f$) is the largest triangular number possible, and @f$ r \ge 0 @f$ the remainder.
 *  @param index Flatten index.
 *  @returns Row (@f$ k @f$) and column (@f$ r @f$) index of lower triangular matrix.
 */
__cuhostdev__ std::array<std::uint64_t, 2> triangular_index(std::uint64_t index) noexcept;

// Random Subset
// -------------

/** @brief Get a random subset of index in a range.
 *  @param num_points Number of points to get.
 *  @param i_max Index of max range.
 *  @param i_min Index of min range.
 */
MERLIN_EXPORTS UIntVec get_random_subset(std::uint64_t num_points, std::uint64_t i_max,
                                         std::uint64_t i_min = 0) noexcept;

}  // namespace merlin

#endif  // MERLIN_UTILS_HPP_
