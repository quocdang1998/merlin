// Copyright 2022 quocdang1998
#include "merlin/utils.hpp"

#include "merlin/logger.hpp"

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// tensor tools
// --------------------------------------------------------------------------------------------------------------------

std::vector<unsigned int> contiguous_strides(const std::vector<unsigned int> & dims, unsigned int element_size) {
    std::vector<unsigned int> contiguous_strides(dims.size(), element_size);
    for (int i = dims.size()-2; i >= 0; i--) {
        contiguous_strides[i] = contiguous_strides[i+1] * dims[i+1];
    }
    return contiguous_strides;
}


unsigned int leap(const std::vector<unsigned int> & index, const std::vector<unsigned int> & strides) {
    if (index.size() != strides.size()) {
        FAILURE("Size of index (%d) and size of strides (%d) are not equal.", index.size(), strides.size());
    }
    unsigned int leap = 0;
    for (int i = 0; i < index.size(); i++) {
        leap += index[i] * strides[i];
    }
    return leap;
}


std::tuple<unsigned int, int> lcseg_and_brindex(const std::vector<unsigned int> & dims,
                                                const std::vector<unsigned int> & strides) {
    // check size of 2 vectors
    if (dims.size() != strides.size()) {
        FAILURE("Size of dims (%d) and size of strides (%d) are not equal.", dims.size(), strides.size());
    }

    // initialize elements
    unsigned int ndim_ = dims.size();
    std::vector<unsigned int> contiguous_strides_ = contiguous_strides(dims, sizeof(float));
    unsigned int longest_contiguous_segment_ = sizeof(float);
    int break_index_ = ndim_ - 1;

    // check if i-th element of strides equals to i-th element of contiguous_strides,
    // break at the element of different index
    for (int i = ndim_-1; i >= 0; i--) {
        if (strides[i] == contiguous_strides_[i]) {
            longest_contiguous_segment_ *= dims[i];
            break_index_--;
        } else {
            break;
        }
    }

    return std::tuple<unsigned int, int>(longest_contiguous_segment_, break_index_);
}

unsigned int ndim_to_contiguous_idx(const std::vector<unsigned int> & index, const std::vector<unsigned int> & dims) {
    if (index.size() != dims.size()) {
        FAILURE("Expect size of index (%d) and size of dims (%d) to be equal.", index.size(), dims.size());
    }
    unsigned int contiguous_index = 0;
    for (int i = 0; i < dims.size(); i++) {
        contiguous_index += dims[i]*index[i];
    }
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

// --------------------------------------------------------------------------------------------------------------------
// job partition utils
// --------------------------------------------------------------------------------------------------------------------



}  // namespace merlin
