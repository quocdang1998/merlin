// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_TPP_
#define MERLIN_UTILS_TPP_

#include <algorithm>  // std::min, std::max

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// NdData tools
// -------------------------------------------------------------------------------------------------------------------------

// Copy each segment from source to destination
template <class CopyFunction>
void array_copy(NdData * dest, const NdData * src, CopyFunction copy) {
    // check if shape vector are the same
    if (src->ndim() != dest->ndim()) {
        FAILURE(std::invalid_argument, "Cannot copy array of different ndim (%u to %u).\n", src->ndim(), dest->ndim());
    }
    unsigned long int ndim = src->ndim();
    for (int i = 0; i < ndim; i++) {
        if (dest->shape()[i] < src->shape()[i]) {
            FAILURE(std::invalid_argument, "Expected shape at index %d of source (%d) smaller or equal destination (%d).\n",
                    i, src->shape()[i], dest->shape()[i]);
        }
    }
    intvec shape(src->shape());

    // longest contiguous segment and break index of the source
    unsigned long int src_lcs, des_lcs;
    long int src_bridx, des_bridx;
    std::tie(src_lcs, src_bridx) = lcseg_and_brindex(shape, src->strides());
    std::tie(des_lcs, des_bridx) = lcseg_and_brindex(shape, dest->strides());
    unsigned long int longest_contiguous_segment = std::min(src_lcs, des_lcs);
    long int break_index = std::max(src_bridx, des_bridx);

    // copy each longest contiguous segment through the copy function
    if (break_index == -1) {  // original tensor is perfectly contiguous
        copy(dest->data(), src->data(), longest_contiguous_segment);
    } else {  // memcpy each longest_contiguous_segment
        // initilize index vector
        intvec index(ndim, 0);
        while (true) {
            // calculate ptr to each segment
            unsigned int src_leap = inner_prod(index, src->strides());
            uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src->data()) + src_leap;
            unsigned int des_leap = inner_prod(index, dest->strides());
            uintptr_t des_ptr = reinterpret_cast<uintptr_t>(dest->data()) + des_leap;
            // copy the segment
            copy(reinterpret_cast<float *>(des_ptr), reinterpret_cast<float *>(src_ptr), longest_contiguous_segment);
            // increase index at break index by 1 and carry surplus if index value exceeded shape
            index[break_index] += 1;
            int update_dim = break_index;
            while ((index[update_dim] == shape[update_dim]) && (update_dim > 0)) {
                index[update_dim] = 0;
                index[--update_dim] += 1;
            }
            // break if all element is looped
            if (index[0] == shape[0]) {
                break;
            }
        }
    }
}

}  // namespace merlin

#endif  // MERLIN_UTILS_TPP_
