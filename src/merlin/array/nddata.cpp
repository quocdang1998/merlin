// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// NdData
// --------------------------------------------------------------------------------------------------------------------

// Member initialization for C++ interface
array::NdData::NdData(float * data, std::uint64_t ndim,
                      const intvec & shape, const intvec & strides) : ndim_(ndim), shape_(shape), strides_(strides) {
    this->data_ = data;
    if (shape.size() != ndim) {
        FAILURE(std::range_error, "Expected size of shape (%u) equals to ndim (%u).\n", shape.size(), ndim);
    } else if (strides.size() != ndim) {
        FAILURE(std::range_error, "Expected size of strides (%u) equals to ndim (%u).\n", strides.size(), ndim);
    }
}

// Constructor from Numpy np.array
array::NdData::NdData(float * data, std::uint64_t ndim, const std::uint64_t * shape, const std::uint64_t * strides) {
    this->ndim_ = ndim;
    this->data_ = data;
    this->shape_ = intvec(shape, shape + ndim);
    this->strides_ = intvec(strides, strides + ndim);
}

// Constructor from a slice
array::NdData::NdData(const array::NdData & whole, std::initializer_list<array::Slice> slices) {
    // check size
    if (slices.size() != whole.ndim_) {
        CUHDERR(std::invalid_argument, "Dimension of Slices and NdData not compatible (expected %u, got %u).\n",
                static_cast<unsigned int>(whole.ndim_), static_cast<unsigned int>(slices.size()));
    }
    // create result
    const array::Slice * slice_data = slices.begin();
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(whole.data_);
    std::uintptr_t result_ndim = 0;
    intvec new_shape(whole.ndim_, 0);
    intvec new_strides(whole.ndim_, 0);
    for (int i = 0; i < whole.ndim_; i++) {
        auto [offset, shape, stride] = slice_data[i].slice_on(whole.shape_[i], whole.strides_[i]);
        data_ptr += offset;
        if (shape != 1) {
            new_shape[result_ndim] = shape;
            new_strides[result_ndim] = stride;
            result_ndim++;
        }
    }
    // finalize
    this->data_ = reinterpret_cast<float *>(data_ptr);
    this->ndim_ = (result_ndim == 0) ? 1 : result_ndim;
    this->shape_ = (result_ndim == 0) ? intvec({1}) : intvec(new_shape.data(), result_ndim);
    this->strides_ = (result_ndim == 0) ? intvec({sizeof(float)}) : intvec(new_strides.data(), result_ndim);
}

}  // namespace merlin
