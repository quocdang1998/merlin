// Copyright 2022 quocdang1998
#include "merlin/array.hpp"

#include <cstdlib>  // div_t, div
#include <cstring>  // std::memcpy
#include <utility>  // std::move

#include "merlin/parcel.hpp"  // merlin::Parcel
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::inner_prod, merlin::contiguous_strides, merlin::array_copy

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Array (CPU)
// --------------------------------------------------------------------------------------------------------------------

// Constructor Array of one element
Array::Array(float value) {
    // allocate data
    this->data_ = new float[1];
    this->data_[0] = value;

    // set metadata
    this->ndim_ = 1;
    this->strides_ = intvec({sizeof(float)});
    this->shape_ = intvec({1});
    this->force_free = true;
}

// Construct empty Array from shape vector
Array::Array(std::initializer_list<unsigned long int> shape) {
    // initilaize ndim and shape
    this->ndim_ = shape.size();
    this->shape_ = intvec(shape);
    // calculate strides
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // initialize data
    this->data_ = new float[this->size()];
    // other meta data
    this->force_free = true;
}

// Construct Array from Numpy array
Array::Array(float * data, unsigned long int ndim,
               unsigned long int * shape, unsigned long int * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->shape_ = intvec(shape, shape + ndim);
    this->strides_ = intvec(strides, strides + ndim);
    this->force_free = copy;

    // copy / assign data
    if (copy) {  // copy data
        // allocate a new tensor
        this->data_ = new float[this->size()];
        // reform the stride tensor (force into C shape)
        this->strides_ = contiguous_strides(this->shape_, sizeof(float));
        // copy data from old tensor to new tensor (optimized with memcpy)
        NdData src(data, ndim, shape, strides);
        array_copy(dynamic_cast<NdData *>(this), &src, std::memcpy);
    } else {
        this->data_ = data;
    }
}

// Copy constructor
Array::Array(const Array & src) : NdData(src) {
    // copy / initialize meta data
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    this->force_free = true;
    // copy data
    this->data_ = new float[this->size()];
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&src), std::memcpy);
}

// Copy assignment
Array & Array::operator=(const Array & src) {
    // copy / initialize meta data
    this->strides_ = contiguous_strides(this->shape_, sizeof(float));
    // free current data
    if (this->force_free) {
        delete[] this->data_;
    }
    this->force_free = true;
    // copy data
    this->data_ = new float[this->size()];
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&src), std::memcpy);
    return *this;
}

// Move constructor
Array::Array(Array && src) : NdData(src) {
    // disable force_free of the source
    this->force_free = src.force_free;
    src.force_free = false;
    // nullify source data
    src.data_ = NULL;
}

// Move assignment
Array & Array::operator=(Array && src) {
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
Array::iterator Array::begin(void) {
    this->begin_ = intvec(this->ndim_, 0);
    this->end_ = intvec(this->ndim_, 0);
    this->end_[0] = this->shape_[0];
    return Array::iterator(this->begin_, this->shape_);
}

// End iterator
Array::iterator Array::end(void) {
    return Array::iterator(this->end_, this->shape_);
}

// Get value operator
float & Array::operator[] (const intvec & idx) {
    unsigned long int leap = inner_prod(idx, this->strides_);
    uintptr_t data_ptr = reinterpret_cast<uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<float *>(data_ptr));
}

// Copy data from GPU array
#ifndef __MERLIN_CUDA__
void sync_from_gpu(const Parcel & gpu_array, uintptr_t stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}
#endif  // __MERLIN_CUDA__

// Destructor
Array::~Array(void) {
    if (this->force_free) {
        delete[] this->data_;
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Array::iterator
// --------------------------------------------------------------------------------------------------------------------

// Pre-increment operator
Array::iterator & Array::iterator::operator++(void) {
    this->index_[this->index_.size() - 1]++;
    unsigned long int current_dim = this->index_.size() - 1;
    intvec & shape = this->shape();
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
Array::iterator Array::iterator::operator++(int) {
    return ++(*this);
}

// Comparison iterator
bool operator!= (const Array::iterator & left, const Array::iterator & right) {
    // check if 2 iterators comes from the same tensor
    if (left.shape_ != right.shape_) {
        FAILURE(std::invalid_argument, "2 iterators are not comming from the same tensor.\n");
    }
    // compare index of each iterator
    unsigned long int length = left.index().size();
    for (int i = 0; i < length; i++) {
        if (left.index_[i] != right.index_[i]) {
            return true;
        }
    }
    return false;
}

// Update index after a manual modification (deprecated)
void Array::iterator::update(void) {
    // detect dimensions having index bigger than dim
    unsigned long int current_dim = this->index_.size();
    intvec & shape = this->shape();
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
