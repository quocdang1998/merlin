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
// Array
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
             const unsigned long int * shape, const unsigned long int * strides, bool copy) {
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
    this->NdData::operator=(src);
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
    // copy meta data
    this->NdData::operator=(src);
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
    return Array::iterator(this->begin_, *this);
}

// End iterator
Array::iterator Array::end(void) {
    return Array::iterator(this->end_, *this);
}

// Get value operator
float & Array::operator[] (const intvec & index) {
    unsigned long int leap = inner_prod(index, this->strides_);
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

}  // namespace merlin
