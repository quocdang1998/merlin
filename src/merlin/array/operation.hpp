// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_OPERATION_HPP_
#define MERLIN_ARRAY_OPERATION_HPP_

#include <cstdint>  // std::uint64_t
#include <string>   // std::string
#include <utility>  // std::pair

#include <iostream>

#include "merlin/array/nddata.hpp"    // merlin::array::NdData
#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/settings.hpp"        // merlin::Index

namespace merlin::array {

/** @brief Calculate C-contiguous stride vector.
 *  @details Get stride vector from dims vector and size of one element as if the merlin::array::NdData is
 *  C-contiguous.
 *  @param shape Shape vector.
 *  @param ndim Number of dimensions.
 *  @param element_size Size on one element.
 */
constexpr Index contiguous_strides(const Index & shape, std::uint64_t ndim, std::uint64_t element_size) {
    Index c_strides;
    c_strides[ndim - 1] = element_size;
    for (std::int64_t i = ndim - 2; i >= 0; i--) {
        c_strides[i] = shape[i + 1] * c_strides[i + 1];
    }
    return c_strides;
}

/** @brief Calculate the number of bytes to jump to get element at a given C-contiguous index.
 *  @details This function is equivalent to the successive calls of `merlin::contiguous_to_ndim_idx` and
 *  `merlin::inner_prod`, but it does not require any memory allocation.
 *  @param index C-contiguous index.
 *  @param shape Shape vector.
 *  @param strides Strides vector.
 *  @param ndim Number of dimensions.
 */
__cuhostdev__ std::uint64_t get_leap(std::uint64_t index, const Index & shape, const Index & strides,
                                     std::uint64_t ndim) noexcept;

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
constexpr std::pair<std::uint64_t, std::int64_t> lcseg_and_brindex(const Index & shape, const Index & strides,
                                                                   std::uint64_t ndim) {
    std::uint64_t longest_contiguous_segment = sizeof(double);
    std::int64_t break_index = ndim - 1;
    for (std::int64_t i = ndim - 1; i > 0; i--) {
        if (strides[i] != longest_contiguous_segment) {
            break;
        }
        longest_contiguous_segment *= shape[i];
        break_index -= 1;
    }
    if (strides[0] == longest_contiguous_segment) {
        longest_contiguous_segment *= shape[0];
        break_index = -1;
    }
    return {longest_contiguous_segment, break_index};
}

/** @brief Copy data from an merlin::array::NdData to another.
 *  @details This function allows user to choose the copy function (for example, ``std::memcpy``, or ``cudaMemcpy``).
 *  @tparam CopyFunction Function copy an array to another, having the prototype of
 *  ``void WriteFunction(void * dest, const void * src, std::size_t size_in_bytes)``.
 *  @param dest Pointer to destination merlin::array::NdData.
 *  @param src Pointer to source merlin::array::NdData.
 *  @param copy Name of the copy function.
 */
template <class CopyFunction>
void copy(array::NdData * dest, const array::NdData * src, CopyFunction copy);

/** @brief Fill all array with a given value.
 *  @tparam WriteFunction Function writing from a CPU array to the target pointer data, having the prototype of
 *  ``void WriteFunction(void * dest, const void * src, std::size_t size_in_bytes)``.
 *  @param target Target array to fill.
 *  @param fill_value Value to fill the array.
 *  @param write_engine Name of the function writing to array.
 *  @param buffer Size of the buffer to write to the array
 */
template <class CopyFunction>
void fill(array::NdData * target, double fill_value, CopyFunction write_engine,
          std::uint64_t buffer = 1024 * sizeof(double));

/** @brief String representation of an array.
 *  @param target Target array to print.
 *  @param nametype Typename of the array to be printed (``Array``, ``Parcel``, ``Stock`` or ``NdData``).
 *  @param first_call Check if the print is called on the first level (highest dimension).
 */
template <class NdArray>
std::string print(const NdArray * target, const std::string & nametype, bool first_call);

}  // namespace merlin::array

#include "merlin/array/operation.tpp"

#endif  // MERLIN_ARRAY_OPERATION_HPP_
