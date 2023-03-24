// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include <string>  // std::string

#include "merlin/cuda_decorator.hpp"  // __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_HOSTDEV_EXPORTS
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin {

// System
// ------

/** @brief Get process ID in form of a string.*/
std::string get_current_process_id(void);

/** @brief Get current time in form of a string.*/
std::string get_time(void);

// CUDA kernel
// -----------

/** @brief Calculate number of block in a grid give the size of each block and the size of the loop.*/
__cuhostdev__ constexpr std::uint64_t get_block_count(std::uint64_t block_size, std::uint64_t loop_size) {
    return (loop_size + block_size - 1) / block_size;
}

#ifdef __NVCC__

/** @brief Flatten index of the current thread in block.*/
__cudevice__ constexpr std::uint64_t flatten_thread_index(void) {
    std::uint64_t result = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    return result;
}

/** @brief Size of thread block.*/
__cudevice__ constexpr std::uint64_t size_of_block(void) {
    std::uint64_t result = blockDim.x*blockDim.y*blockDim.z;
    return result;
}

/** @brief Flatten index of the current grid block.*/
__cudevice__ constexpr std::uint64_t flatten_block_index(void) {
    std::uint64_t result = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    return result;
}

/** @brief Flatten index of the current thread block.*/
__cudevice__ constexpr std::uint64_t flatten_kernel_index(void) {
    std::uint64_t index_in_block = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    std::uint64_t index_of_block = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    std::uint64_t size_of_one_block = blockDim.x*blockDim.y*blockDim.z;
    return index_of_block*size_of_one_block + index_in_block;
}

#endif  // __NVCC__

// Multi-dimensional Index
// -----------------------

/** @brief Product of entries of a vector.
 *  @details Return product of all elements of the vector.
 *  @param v Vector.
 *  @note This function returns ``1`` if ``v`` has zero size.
 */
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS std::uint64_t prod_elements(const intvec & v);

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
 *  @param data_ptr Pointer to result data. If the value is ``nullptr``, new instance is allocated.
 *  @return merlin::intvec of n-dimensional index.
 */
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS intvec contiguous_to_ndim_idx(std::uint64_t index, const intvec & shape,
                                                                   std::uint64_t * data_ptr = nullptr);

/** @brief Increase an n-dimensional index by one unit.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return Lowest changed dimension.
 */
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS std::uint64_t increment_index(intvec & index, const intvec & shape);

/** @brief Decrease an n-dimensional index by one unit.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return Lowest changed dimension.
 */
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS std::uint64_t decrement_index(intvec & index, const intvec & shape);

// Sparse Grid
// -----------

/** @brief Get size of a sub-grid given its level vector.*/
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS std::uint64_t calc_subgrid_size(const intvec & level_vector) noexcept;

/** @brief Get shape of Cartesian subgrid corresponding to a level vector.*/
__cuhostdev__ MERLIN_HOSTDEV_EXPORTS intvec get_level_shape(const intvec & level_vector);





/** @brief Index of nodes belonging to a level of a 1D grid.
 *  @param level Level to get index.
 *  @param size Size of 1D grid level.
 */
// __cuhostdev__ intvec hiearchical_index(std::uint64_t level, std::uint64_t size);  // x1

}  // namespace merlin

#endif  // MERLIN_UTILS_HPP_
