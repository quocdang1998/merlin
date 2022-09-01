// Copyright 2022 quocdang1998
#include "merlin/utils.hpp"

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// Miscellaneous utils
// -------------------------------------------------------------------------------------------------------------------------

// Inner product
#ifndef __MERLIN_CUDA__
unsigned long int inner_prod(const intvec & v1, const intvec & v2) {
    if (v1.size() != v2.size()) {
        FAILURE(std::invalid_argument, "Size of v1 (%d) and size of v2 (%d) are not equal.\n", v1.size(), v2.size());
    }
    // calculate inner product
    unsigned long int inner_product = 0;
    for (int i = 0; i < v1.size(); i++) {
        inner_product += v1[i] * v2[i];
    }
    return inner_product;
}
#endif  // __MERLIN_CUDA__

// -------------------------------------------------------------------------------------------------------------------------
// NdData tools
// -------------------------------------------------------------------------------------------------------------------------

// C-Contiguous strides from shape vector
intvec contiguous_strides(const intvec & shape, unsigned long int element_size) {
    intvec c_strides = intvec(shape.size(), element_size);
    for (int i = c_strides.size()-2;  i >= 0; i--) {
        c_strides[i] = shape[i+1] * c_strides[i+1];
    }
    return c_strides;
}

#ifndef __MERLIN_CUDA__
// Convert n-dimensional index to C-contiguous index
unsigned long int ndim_to_contiguous_idx(const intvec & index, const intvec & shape) {
    return inner_prod(index, shape);
}

// Convert C-contiguous index to n-dimensional index
intvec contiguous_to_ndim_idx(unsigned long int index, const intvec & shape) {
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
#endif  // __MERLIN_CUDA__

// Longest contiguous segment and break index
std::tuple<unsigned long int, long int> lcseg_and_brindex(const intvec & shape, const intvec & strides) {
    // check size of 2 vectors
    if (shape.size() != strides.size()) {
        FAILURE(std::runtime_error, "Size of shape (%d) and size of strides (%d) are not equal.\n",
                shape.size(), strides.size());
    }

    // initialize elements
    unsigned long int ndim_ = shape.size();
    intvec contiguous_strides_ = contiguous_strides(shape, sizeof(float));
    unsigned long int longest_contiguous_segment_ = sizeof(float);
    long int break_index_ = ndim_ - 1;

    // check if i-th element of strides equals to i-th element of contiguous_strides,
    // break at the element of different index
    for (int i = ndim_-1; i >= 0; i--) {
        if (strides[i] == contiguous_strides_[i]) {
            longest_contiguous_segment_ *= shape[i];
            break_index_--;
        } else {
            break;
        }
    }

    return std::tuple<unsigned long int, long int>(longest_contiguous_segment_, break_index_);
}



// -------------------------------------------------------------------------------------------------------------------------
// job partition utils
// -------------------------------------------------------------------------------------------------------------------------



}  // namespace merlin
