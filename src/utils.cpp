// Copyright 2022 quocdang1998
#include "merlin/utils.hpp"

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Miscellaneous utils
// --------------------------------------------------------------------------------------------------------------------

// Inner product
unsigned long int inner_prod(const intvec & v1, const intvec & v2) {
    // check size of 2 vectors
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

// --------------------------------------------------------------------------------------------------------------------
// NdData tools
// --------------------------------------------------------------------------------------------------------------------

// C-Contiguous strides from shape vector
intvec contiguous_strides(const intvec & shape, unsigned long int element_size) {
    intvec c_strides = intvec(shape.size(), element_size);
    for (int i = c_strides.size()-2;  i >= 0; i--) {
        c_strides[i] = shape[i+1] * c_strides[i+1];
    }
    return c_strides;
}

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
/*
unsigned int ndim_to_contiguous_idx(const std::vector<unsigned int> & index, const std::vector<unsigned int> & dims) {
    unsigned int contiguous_index = inner_prod<unsigned int>(index, dims);
    return contiguous_index;
}


std::vector<std::vector<unsigned int>> contiguous_to_ndim_idx(const std::vector<unsigned int> & index,
                                                              const std::vector<unsigned int> & dims) {
    // create prod vector (cumulative product from the last element)
    std::vector<unsigned int> prod_(dims.size(), 1);
    for (int i = dims.size()-2; i >= 0; i--) {
        prod_[i] = prod_[i+1]*dims[i+1];
    }
    // create n-dim index vector for each C-contiguous index value
    std::vector<std::vector<unsigned int>> result;
    for (int i = 0; i < index.size(); i++) {
        std::vector<unsigned int> nd_idx(dims.size(), 0);
        for (int j = 0; j < dims.size(); j++) {
            nd_idx[j] = (index[i] / prod_[j]) % dims[j];
        }
        result.push_back(nd_idx);
    }
    return result;
}
*/
// --------------------------------------------------------------------------------------------------------------------
// job partition utils
// --------------------------------------------------------------------------------------------------------------------



}  // namespace merlin
