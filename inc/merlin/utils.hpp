// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_HPP_
#define MERLIN_UTILS_HPP_

#include <tuple>  // std::tuple

#include "merlin/nddata.hpp"  // merlin::NdData
#include "merlin/decorator.hpp"  // __cuhost__, __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

// Miscellaneous utils
// -------------------

/** @brief Inner product of 2 index vectors.
 *  @details Return convolution product / scalar product of 2 vectors.
 *  @param v1 First vector.
 *  @param v2 Second vector.
 */
__cuhostdev__ inline unsigned long int inner_prod(const intvec & v1, const intvec & v2) {
    // check size of 2 vectors
    #ifndef __CUDA_ARCH__
    if (v1.size() != v2.size()) {
        FAILURE(std::invalid_argument, "Size of v1 (%u) and size of v2 (%u) are not equal.\n", v1.size(), v2.size());
    }
    #endif  // __CUDA_ARCH__
    // calculate inner product
    unsigned long int inner_product = 0;
    for (int i = 0; i < v1.size(); i++) {
        inner_product += v1[i] * v2[i];
    }
    return inner_product;
}

// NdData tools
// ------------

/** @brief Calculate C-contiguous stride vector.
 *  @details Get stride vector from dims vector and size of one element as if the NdData is C-contiguous.
 *  @param shape Shape vector.
 *  @param element_size Size on one element.
 */
MERLIN_EXPORTS intvec contiguous_strides(const intvec & shape, unsigned long int element_size);

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
MERLIN_EXPORTS std::tuple<unsigned long int, long int> lcseg_and_brindex(const intvec & shape, const intvec & strides);

/** @brief Copy data from an NdData to another.
 *  @details This function allows user to choose the copy function (for example, std::memcpy, or cudaMemcpy).
 *  @tparam CopyFunction Function copy an array to another. This function must take exactly 3 arguments:
 *  destination pointer, source pointer and length of copied memory in bytes.
 *  @param dest Pointer to destination NdData.
 *  @param src Pointer to source NdData.
 *  @param copy Name of the copy function.
 */
template <class CopyFunction>
void array_copy(NdData * dest, const NdData * src, CopyFunction copy);

/** @brief Convert n-dimensional index to C-contiguous index.
 *  @param index Multi-dimensional index.
 *  @param shape Shape vector.
 *  @return C-contiguous index as an ``unsigned long integer``.
 */
__cuhostdev__ inline unsigned long int ndim_to_contiguous_idx(const intvec & index, const intvec & shape) {
    return inner_prod(index, shape);
}

/** @brief Convert C-contiguous index to n-dimensional index.
 *  @param index C-contiguous index.
 *  @param shape Shape vector.
 *  @return merlin::intvec of n-dimensional index.
 */
__cuhostdev__ inline intvec contiguous_to_ndim_idx(unsigned long int index, const intvec & shape) {
    // calculate index vector
    intvec index_(shape.size());
    unsigned long int cum_prod;
    for (int i = shape.size()-1; i >= 0; i--) {
        if (i == shape.size()-1) {
            cum_prod = 1;
        } else {
            cum_prod = cum_prod * shape[i+1];
        }
        index_[i] = (index / cum_prod) % shape[i];
    }
    return index_;
}

// Parcel tools
// ------------

#ifdef __NVCC__

__cuhostdev__ inline int get_current_device(void) {
    int current_device;
    cudaGetDevice(&current_device);
    return current_device;
}

#endif  // __NVCC__

#ifdef COMMENT

std::vector<std::vector<unsigned int>> partition_list(unsigned int length, unsigned int n_segment);
#endif

}  // namespace merlin

#include "merlin/utils.tpp"

#endif  // MERLIN_UTILS_HPP_
