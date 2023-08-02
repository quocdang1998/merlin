// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_TPP_
#define MERLIN_UTILS_TPP_

#include <algorithm>  // std::min, std::max

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"   // merlin::inner_prod

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// NdData tools
// ---------------------------------------------------------------------------------------------------------------------

// Copy each segment from source to destination
template <class CopyFunction>
void array::array_copy(array::NdData * dest, const array::NdData * src, CopyFunction copy) {
    // check if shape vector are the same
    if (src->ndim() != dest->ndim()) {
        FAILURE(std::invalid_argument, "Cannot copy array of different ndim (%u to %u).\n", src->ndim(), dest->ndim());
    }
    std::uint64_t ndim = src->ndim();
    if (dest->shape() != src->shape()) {
        FAILURE(std::invalid_argument, "Expected shape of source equals shape of destination.\n");
    }
    intvec shape(src->shape());
    // trivial case: size zero
    if (ndim == 0) {
        return;
    }
    // longest contiguous segment and break index of the source
    std::uint64_t src_lcs, des_lcs;
    std::int64_t src_bridx, des_bridx;
    std::tie(src_lcs, src_bridx) = array::lcseg_and_brindex(shape, src->strides());
    std::tie(des_lcs, des_bridx) = array::lcseg_and_brindex(shape, dest->strides());
    std::uint64_t longest_contiguous_segment = std::min(src_lcs, des_lcs);
    std::int64_t break_index = std::max(src_bridx, des_bridx);

    // copy each longest contiguous segment through the copy function
    if (break_index == -1) {  // original arrays are perfectly contiguous
        copy(dest->data(), src->data(), longest_contiguous_segment);
    } else {  // memcpy each longest_contiguous_segment
        // initilize index vector
        intvec index(ndim, 0);
        while (true) {
            // calculate ptr to each segment
            std::uint64_t src_leap = inner_prod(index, src->strides());
            std::uintptr_t src_ptr = reinterpret_cast<std::uintptr_t>(src->data()) + src_leap;
            std::uint64_t des_leap = inner_prod(index, dest->strides());
            std::uintptr_t des_ptr = reinterpret_cast<std::uintptr_t>(dest->data()) + des_leap;
            // copy the segment
            copy(reinterpret_cast<double *>(des_ptr), reinterpret_cast<double *>(src_ptr), longest_contiguous_segment);
            // increase index at break index by 1 and carry surplus if index value exceeded shape
            index[break_index] += 1;
            std::int64_t update_dim = break_index;
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
