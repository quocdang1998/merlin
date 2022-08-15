// Copyright 2022 quocdang1998
#include "merlin/tensor.hpp"

#include <cstdlib>  // div_t, div
#include <cstring>  // std::memcpy
#include <utility>  // std::move

#include "merlin/parcel.hpp"  // Parcel
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // inner_prod, contiguous_strides, array_copy

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Tensor (CPU)
// --------------------------------------------------------------------------------------------------------------------

// Constructor Tensor of one element
Tensor::Tensor(float value) {
    // allocate data
    this->data_ = new float[1];
    this->data_[0] = value;

    // set metadata
    this->ndim_ = 1;
    this->strides_ = std::vector<unsigned int>(1, sizeof(float));
    this->shape_ = std::vector<unsigned int>(1, 1);
    this->force_free = true;
}

// Construct empty Tensor from shape vector
Tensor::Tensor(std::initializer_list<unsigned int> shape) {
    // initilaize ndim and shape
    this->ndim_ = shape.size();
    this->shape_ = std::vector<unsigned int>(shape);
    // calculate strides
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // initialize data
    this->data_ = new float[this->size()];
    // other meta data
    this->force_free = true;
}

// Construct Tensor from Numpy array
Tensor::Tensor(float * data, unsigned int ndim, unsigned int * shape, unsigned int * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->shape_ = std::vector<unsigned int>(shape, shape + ndim);
    this->strides_ = std::vector<unsigned int>(strides, strides + ndim);
    this->force_free = copy;

    // copy / assign data
    if (copy) {  // copy data
        // allocate a new tensor
        this->data_ = new float[this->size()];
        // reform the stride tensor (force into C shape)
        this->strides_ = contiguous_strides(this->shape_, sizeof(float));
        // copy data from old tensor to new tensor (optimized with memcpy)
        Array src(data, ndim, shape, strides);
        array_copy(dynamic_cast<Array *>(this), &src, std::memcpy);
    } else {
        this->data_ = data;
    }
}

// Copy constructor
Tensor::Tensor(const Tensor & src) : Array(src) {
    // copy / initialize meta data
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    this->force_free = true;
    // copy data
    this->data_ = new float[this->size()];
    array_copy(dynamic_cast<Array *>(this), dynamic_cast<const Array *>(&src), std::memcpy);
}

// Copy assignment
Tensor & Tensor::operator=(const Tensor & src) {
    // copy / initialize meta data
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // free current data
    if (this->force_free) {
        delete[] this->data_;
    }
    this->force_free = true;
    // copy data
    this->data_ = new float[this->size()];
    array_copy(dynamic_cast<Array *>(this), dynamic_cast<const Array *>(&src), std::memcpy);
    return *this;
}

// Move constructor
Tensor::Tensor(Tensor && src) : Array(src) {
    // disable force_free of the source
    this->force_free = src.force_free;
    src.force_free = false;
    // nullify source data
    src.data_ = NULL;
}

// Move assignment
Tensor & Tensor::operator=(Tensor && src) {
    // disable force_free of the source and free current data
    if (this->force_free) {
        delete[] this->data_;
    }
    this->force_free = src.force_free;
    src.force_free = false;
    // move data
    src.data_ = NULL;
    return *this;
}

// Begin iterator
Tensor::iterator Tensor::begin(void) {
    this->begin_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_[0] = this->shape_[0];
    return Tensor::iterator(this->begin_, this->shape_);
}

// End iterator
Tensor::iterator Tensor::end(void) {
    return Tensor::iterator(this->end_, this->shape_);
}

// Get value operator
float & Tensor::operator[] (const std::vector<unsigned int> & index) {
    unsigned int leap = inner_prod<unsigned int>(index, this->strides_);
    uintptr_t data_ptr = reinterpret_cast<uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<float *>(data_ptr));
}

// Copy data from GPU array
#ifndef MERLIN_CUDA_
void sync_from_gpu(const Parcel & gpu_array, uintptr_t stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}
#endif  // MERLIN_CUDA_

// Destructor
Tensor::~Tensor(void) {
    if (this->force_free) {
        delete[] this->data_;
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Tensor::iterator
// --------------------------------------------------------------------------------------------------------------------

// Pre-increment operator
Tensor::iterator & Tensor::iterator::operator++(void) {
    this->index_[this->index_.size() - 1]++;
    unsigned int current_dim = this->index_.size() - 1;
    std::vector<unsigned int> & shape = this->shape();
    while (this->index_[current_dim] >= shape[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == shape[current_dim]) {
                break;
            } else {
                FAILURE(std::out_of_range, "Maximum size reached, cannot add more.\n");
            }
        }
        this->index_[current_dim] = 0;
        this->index_[--current_dim] += 1;
    }
    return *this;
}

// Post-increment operator
Tensor::iterator Tensor::iterator::operator++(int) {
    return ++(*this);
}

// Comparison iterator
bool operator!= (const Tensor::iterator & left, const Tensor::iterator & right) {
    // check if 2 iterators comes from the same tensor
    if (left.shape_ != right.shape_) {
        FAILURE(std::invalid_argument, "2 iterators are not comming from the same tensor.\n");
    }
    // compare index of each iterator
    unsigned int length = left.index().size();
    for (int i = 0; i < length; i++) {
        if (left.index_[i] != right.index_[i]) {
            return true;
        }
    }
    return false;
}

// Update index after a manual modification (deprecated)
void Tensor::iterator::update(void) {
    // detect dimensions having index bigger than dim
    unsigned int current_dim = this->index_.size();
    std::vector<unsigned int>& shape = this->shape();
    for (int i = this->index_.size() - 1; i >= 0; i--) {
        if (this->index_[i] >= shape[i]) {
            current_dim = i;
            break;
        }
    }
    if (current_dim == this->index_.size()) {  // no update needed
        return;
    }
    // carry the surplus to the dimensions with bigger strides
    while (this->index_[current_dim] >= shape[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == shape[current_dim]) {
                break;
            } else {
                FAILURE(std::out_of_range, "Maximum size reached, cannot add more.\n");
            }
        }
        div_t carry = div(static_cast<int>(this->index_[current_dim]), static_cast<int>(shape[current_dim]));
        this->index_[current_dim] = carry.rem;
        this->index_[--current_dim] += carry.quot;
    }
}

}  // namespace merlin
