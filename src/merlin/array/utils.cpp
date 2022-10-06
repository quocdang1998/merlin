// Copyright 2022 quocdang1998
#include "merlin/array/utils.hpp"

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// -------------------------------------------------------------------------------------------------------------------------
// NdData tools
// -------------------------------------------------------------------------------------------------------------------------

// C-Contiguous strides from shape vector
intvec contiguous_strides(const intvec & shape, std::uint64_t element_size) {
    intvec c_strides = intvec(shape.size(), element_size);
    for (int i = c_strides.size()-2;  i >= 0; i--) {
        c_strides[i] = shape[i+1] * c_strides[i+1];
    }
    return c_strides;
}

// Longest contiguous segment and break index
std::tuple<std::uint64_t, std::int64_t> lcseg_and_brindex(const intvec & shape, const intvec & strides) {
    // check size of 2 vectors
    if (shape.size() != strides.size()) {
        FAILURE(std::runtime_error, "Size of shape (%d) and size of strides (%d) are not equal.\n",
                shape.size(), strides.size());
    }

    // initialize elements
    std::uint64_t ndim_ = shape.size();
    intvec contiguous_strides_ = contiguous_strides(shape, sizeof(float));
    std::uint64_t longest_contiguous_segment_ = sizeof(float);
    std::int64_t break_index_ = ndim_ - 1;

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

    return std::tuple<std::uint64_t, std::int64_t>(longest_contiguous_segment_, break_index_);
}

// -------------------------------------------------------------------------------------------------------------------------
// job partition utils
// -------------------------------------------------------------------------------------------------------------------------



}  // namespace merlin
