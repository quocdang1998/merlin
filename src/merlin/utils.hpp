// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include <array>   // std::array
#include <string>  // std::string

#include "merlin/cuda_interface.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/vector.hpp"          // merlin::Vector

namespace merlin {

// System
// ------

/** @brief Get process ID in form of a string.*/
std::string get_current_process_id(void);

/** @brief Get current time in form of a string.
 *  @details Return datetime in form of ``{year}-{month}-{day}_{hour}:{minute}:{second}``.
 */
std::string get_time(void);

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
__cudevice__ constexpr std::uint64_t flatten_thread_index(void) {
    std::uint64_t result = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    return result;
}

/** @brief Size of block.
 *  @details Get the number of threads in the current block.
 */
__cudevice__ constexpr std::uint64_t size_of_block(void) {
    std::uint64_t result = blockDim.x * blockDim.y * blockDim.z;
    return result;
}

/** @brief Block index in grid.
 *  @details Get the three-dimensional flattened index of the current block in the current grid.
 */
__cudevice__ constexpr std::uint64_t flatten_block_index(void) {
    std::uint64_t result = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    return result;
}

/** @brief Thread index in grid.
 *  @details Get the index of the current thread in the current grid.
 */
__cudevice__ constexpr std::uint64_t flatten_kernel_index(void) {
    std::uint64_t index_in_block = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    std::uint64_t index_of_block = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    std::uint64_t size_of_one_block = blockDim.x * blockDim.y * blockDim.z;
    return index_of_block * size_of_one_block + index_in_block;
}

#endif  // __NVCC__

// Multi-dimensional Index
// -----------------------

/** @brief Product of entries of a vector.
 *  @details Return product of all elements of the vector.
 *  @param v Vector.
 *  @note This function returns ``1`` if ``v`` has zero size.
 */
__cuhostdev__ std::uint64_t prod_elements(const intvec & v);

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

/** @brief Convert C-contiguous index to n-dimensional index with allocating memory for result.
 *  @param index C-contiguous index.
 *  @param shape Shape vector.
 */
__cuhostdev__ intvec contiguous_to_ndim_idx(std::uint64_t index, const intvec & shape);

/** @brief Convert C-contiguous index to n-dimensional index and save data to a pre-allocated memory.
 *  @param index C-contiguous index.
 *  @param shape Shape vector.
 *  @param data_ptr Pointer to result data.
 */
__cuhostdev__ void contiguous_to_ndim_idx(std::uint64_t index, const intvec & shape, std::uint64_t * data_ptr);

/** @brief Increase an n-dimensional index by one unit.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return Lowest changed dimension.
 */
__cuhostdev__ std::int64_t increment_index(intvec & index, const intvec & shape);

/** @brief Decrease an n-dimensional index by one unit.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return Lowest changed dimension.
 */
__cuhostdev__ std::int64_t decrement_index(intvec & index, const intvec & shape);

// List Division
// -------------

/** @brief Get a list of pointers divided from an original array.
 *  @param original Pointer to the first element of the original array.
 *  @param divider_length Size of each subsequence.
 *  @param data_ptr Pointer to result data. If the value is ``nullptr``, new instance is allocated.
 */
__cuhostdev__ Vector<double *> ptr_to_subsequence(double * original, const intvec & divider_length,
                                                  double ** data_ptr = nullptr);

/** @brief Get index of sequence and index in that sequence of an index in original array.
 *  @param index_full_array Index in original array.
 *  @param divider_length Size of each subsequence dividing the original array.
 *  @returns Index of the list, and index of the element in the list.
 *  @note If the index in full array overpass the last element in the sequence, the function will return the number of
 *  dimension, and the offset with respect to the last element in the sequence.
 */
__cuhostdev__ std::array<std::uint64_t, 2> index_in_subsequence(std::uint64_t index_full_array,
                                                                const intvec & divider_length) noexcept;

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

}  // namespace merlin

#endif  // MERLIN_UTILS_HPP_
