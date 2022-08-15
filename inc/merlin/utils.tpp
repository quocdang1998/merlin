// Copyright 2022 quocdang1998
#ifndef MERLIN_UTILS_TPP_
#define MERLIN_UTILS_TPP_

#include <algorithm>  // std::min, std::max

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Cumulative product
template <typename NumericType>
std::vector<NumericType> cum_prod(const std::vector<NumericType> & values, bool from_right) {
    // assert NumericType
    static_assert(std::is_arithmetic<NumericType>::value, "NumericType must be a numeric type (int, float, etc).");
    // initilize result vector
    std::vector<NumericType> cummulative_product(values);
    // calculate cummulative_product vector
    if (from_right) {
        for (int i = values.size()-2; i >= 0; i--) {
            cummulative_product[i] = cummulative_product[i+1] * values[i];
        }
    } else {
        for (int i = 1; i < values.size(); i++) {
            cummulative_product[i] = cummulative_product[i-1] * values[i];
        }
    }
    return cummulative_product;
}

// Inner product
template <typename NumericType>
NumericType inner_prod(const std::vector<NumericType> & v1, const std::vector<NumericType> & v2) {
    // assert NumericType
    static_assert(std::is_arithmetic<NumericType>::value, "NumericType must be a numeric type (int, float, etc).");
    // check size of 2 vectors
    if (v1.size() != v2.size()) {
        FAILURE(std::invalid_argument, "Size of v1 (%d) and size of v2 (%d) are not equal.\n", v1.size(), v2.size());
    }
    // calculate inner product
    unsigned int inner_product = 0;
    for (int i = 0; i < v1.size(); i++) {
        inner_product += v1[i] * v2[i];
    }
    return inner_product;
}

// Copy each segment from source to destination
template <class CopyFunction>
void array_copy(Array * dest, const Array * src, CopyFunction copy) {
    // check if shape vector are the same
    if (src->ndim() != dest->ndim()) {
        FAILURE(std::invalid_argument, "Cannot copy array of different ndim (%u to %u).\n", src->ndim(), dest->ndim());
    }
    unsigned int ndim = src->ndim();
    for (int i = 0; i < ndim; i++) {
        if (dest->shape()[i] != src->shape()[i]) {
            FAILURE(std::invalid_argument, "Shape at index %d of source (%d) and destination (%d) are different.\n",
                    i, src->shape()[i], dest->shape()[i]);
        }
    }
    std::vector<unsigned int> shape(src->shape());

    // longest contiguous segment and break index of the source
    unsigned int src_lcs, des_lcs;
    int src_bridx, des_bridx;
    std::tie(src_lcs, src_bridx) = lcseg_and_brindex(shape, src->strides());
    std::tie(des_lcs, des_bridx) = lcseg_and_brindex(shape, dest->strides());
    unsigned int longest_contiguous_segment = std::min(src_lcs, des_lcs);
    int break_index = std::max(src_bridx, des_bridx);

    // copy each longest contiguous segment through the copy function
    if (break_index == -1) {  // original tensor is perfectly contiguous
        copy(dest->data(), src->data(), longest_contiguous_segment);
    } else {  // memcpy each longest_contiguous_segment
        // initilize index vector
        std::vector<unsigned int> index(ndim, 0);
        while (true) {
            // calculate ptr to each segment
            unsigned int src_leap = inner_prod<unsigned int>(index, src->strides());
            uintptr_t src_ptr = reinterpret_cast<uintptr_t>(src->data()) + src_leap;
            unsigned int des_leap = inner_prod<unsigned int>(index, dest->strides());
            uintptr_t des_ptr = reinterpret_cast<uintptr_t>(dest->data()) + des_leap;
            // copy the segment
            copy(reinterpret_cast<float *>(des_ptr), reinterpret_cast<float *>(src_ptr),
                 longest_contiguous_segment);
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
