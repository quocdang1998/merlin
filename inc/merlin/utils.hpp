// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include <vector>  // std::vector
#include <tuple>  // std::tuple
#include <type_traits>  // std::is_arithmetic

#include "merlin/array.hpp"  // Array

namespace merlin {

// Miscellaneous utils
// -------------------

/** @brief Calculate cumulative product of a vector.
 *  @details Return a vector \f$\vec{a}\f$ from input vector \f$\vec{v}\f$ so that:
 *  \f[ a_i = \prod_{j \ge i} v_j \text{ or } a_i = \prod_{j \le i} v_j\f]
 *  @tparam NumericType Numeric type.
 *  @param values Vector of values.
 *  @param from_right Calculate the product from right element (\f$j \ge i\f$) or from left element (\f$j \le i\f$).
 *  @return Vector of cumulative product.
 */
template <typename NumericType>
std::vector<NumericType> cum_prod(const std::vector<NumericType> & values, bool from_right = true);

/** @brief Inner product of 2 vectors.
 *  @details Return convolution product / scalar product of 2 vectors.
 *  @tparam NumericType Numeric type.
 *  @param v1 First vector.
 *  @param v2 Second vector.
 */
template <typename NumericType>
NumericType inner_prod(const std::vector<NumericType> & v1, const std::vector<NumericType> & v2);

// Array tools
// -----------

/** @brief Calculate C-contiguous stride vector.
 *  @details Get stride vector from dims vector and size of one element as if the Tensor is C-contiguous.
 *  @param shape Shape vector.
 *  @param element_size Size on one element.
 */
std::vector<unsigned int> contiguous_strides(const std::vector<unsigned int> & shape, unsigned int element_size);

/** @brief Calculate the longest contiguous segment and break index of an tensor.
 *  @details Longest contiguous segment is the length (in bytes) of the longest sub-tensor that is C-contiguous in the
 *  memory.
 *
 *  Break index is the index at which the tensor break.
 *
 *  For exmample, suppose ``A = [[1.0,2.0,3.0],[4.0,5.0,6.0]]``, then ``A[:,::2]`` will have longest contiguous segment
 *  of 4 and break index of 0.
 *  @param shape Shape vector.
 *  @param strides Strides vector.
*/
std::tuple<unsigned int, int> lcseg_and_brindex(const std::vector<unsigned int> & shape,
                                                const std::vector<unsigned int> & strides);

/** @brief Copy data from an Array to another.
 *  @details This function allows user to choose the copy function (for example, std::memcpy, or cudaMemcpy).
 *  @tparam CopyFunction Function copy an array to another. This function must take exactly 3 arguments:
 *  destination pointer, source pointer and length of copied memory in bytes.
 *  @param dest Pointer to destination Array.
 *  @param src Pointer to source Array.
 *  @param copy Name of the copy function.
 */
template <class CopyFunction>
void array_copy(Array * dest, const Array * src, CopyFunction copy);

// Parcel tools
// ------------

#ifdef __NVCC__

__host__ __device__ inline int get_current_device(void) {
    int current_device;
    cudaGetDevice(&current_device);
    return current_device;
}

#endif  // __NVCC__

/** @brief Convert n-dimensional index to C-contiguous index.

    @param index Multi-dimensional index.
    @param dims Size of each dimension.
*/
unsigned int ndim_to_contiguous_idx(const std::vector<unsigned int> & index, const std::vector<unsigned int> & dims);

/** @brief Convert vector of C-contiguous index to vector of n-dimensional index.

    @param index C-contiguous index.
    @param dims Size of each dimension.
*/
std::vector<std::vector<unsigned int>> contiguous_to_ndim_idx(const std::vector<unsigned int> & index,
                                                              const std::vector<unsigned int> & dims);

std::vector<std::vector<unsigned int>> partition_list(unsigned int length, unsigned int n_segment);

}  // namespace merlin

#include "merlin/utils.tpp"

#endif  // MERLIN_UTILS_HPP_
