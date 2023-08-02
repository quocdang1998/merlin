// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_COPY_HPP_
#define MERLIN_ARRAY_COPY_HPP_

#include <cstdint>  // std::uint64_t, std::int64_t
#include <tuple>    // std::tuple

#include "merlin/array/nddata.hpp"    // merlin::array::NdData
#include "merlin/cuda_interface.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"         // MERLIN_EXPORTS
#include "merlin/vector.hpp"          // merlin::intvec

namespace merlin::array {

/** @brief Calculate C-contiguous stride vector.
 *  @details Get stride vector from dims vector and size of one element as if the merlin::array::NdData is
 *  C-contiguous.
 *  @param shape Shape vector.
 *  @param element_size Size on one element.
 */
__cuhostdev__ intvec contiguous_strides(const merlin::intvec & shape, std::uint64_t element_size);

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
__cuhostdev__ std::tuple<std::uint64_t, std::int64_t> lcseg_and_brindex(const merlin::intvec & shape,
                                                                        const merlin::intvec & strides);

/** @brief Copy data from an merlin::array::NdData to another.
 *  @details This function allows user to choose the copy function (for example, ``std::memcpy``, or ``cudaMemcpy``).
 *  @tparam CopyFunction Function copy an array to another. This function must take exactly 3 arguments:
 *  destination pointer, source pointer and length of copied memory in bytes.
 *  @param dest Pointer to destination merlin::array::NdData.
 *  @param src Pointer to source merlin::array::NdData.
 *  @param copy Name of the copy function.
 */
template <class CopyFunction>
void array_copy(array::NdData * dest, const array::NdData * src, CopyFunction copy);

}  // namespace merlin::array

#include "merlin/array/copy.tpp"

#endif  // MERLIN_ARRAY_COPY_HPP_
