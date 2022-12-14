// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <cstdlib>  // div_t, div
#include <cstring>  // std::memcpy
#include <fstream>  // std::ofstream
#include <mutex>  // std::mutex
#include <utility>  // std::move

#include "merlin/array/copy.hpp"  // merlin::array::contiguous_strides, merlin::array::array_copy
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/array/stock.hpp"  // merlin::array::Stock
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx, merlin::inner_prod

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Array
// --------------------------------------------------------------------------------------------------------------------

// Constructor Array of one element
array::Array::Array(float value) {
    // allocate data
    this->data_ = allocate_memory(1);
    this->data_[0] = value;

    // set metadata
    this->ndim_ = 1;
    this->strides_ = intvec({sizeof(float)});
    this->shape_ = intvec({1});
    this->force_free = true;
}

// Construct empty Array from shape vector
array::Array::Array(const intvec & shape) : array::NdData(shape) {
    // initialize data
    this->data_ = allocate_memory(this->size());
    // other meta data
    this->force_free = true;
}

// Construct Array from Numpy array
array::Array::Array(float * data, std::uint64_t ndim,
             const std::uint64_t * shape, const std::uint64_t * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->shape_ = intvec(shape, shape + ndim);
    this->strides_ = intvec(strides, strides + ndim);
    this->force_free = copy;

    // copy / assign data
    if (copy) {  // copy data
        // allocate a new tensor
        this->data_ = allocate_memory(this->size());
        // reform the stride tensor (force into C shape)
        this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
        // copy data from old tensor to new tensor (optimized with memcpy)
        array::NdData src(data, ndim, shape, strides);
        array::array_copy(dynamic_cast<array::NdData *>(this), &src, std::memcpy);
    } else {
        this->data_ = data;
    }
}

// Constructor from a slice
array::Array::Array(const array::Array & whole, const Vector<array::Slice> & slices) :
array::NdData(whole, slices) {
    this->force_free = false;
}

// Copy constructor
array::Array::Array(const array::Array & src) : array::NdData(src) {
    // copy / initialize meta data
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
    this->force_free = true;
    // copy data
    this->data_ = allocate_memory(this->size());
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), std::memcpy);
}

// Copy assignment
array::Array & array::Array::operator=(const array::Array & src) {
    // copy / initialize meta data
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(float));
    // free current data
    if (this->force_free) {
        delete[] this->data_;
    }
    this->force_free = true;
    // copy data
    this->data_ = allocate_memory(this->size());
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), std::memcpy);
    return *this;
}

// Move constructor
array::Array::Array(array::Array && src) : array::NdData(src) {
    // disable force_free of the source
    this->force_free = src.force_free;
    src.force_free = false;
    // nullify source data
    src.data_ = nullptr;
}

// Move assignment
array::Array & array::Array::operator=(array::Array && src) {
    // disable force_free of the source and free current data
    if (this->force_free) {
        free_memory(this->data_, this->size());
    }
    // copy meta data
    this->array::NdData::operator=(src);
    this->force_free = src.force_free;
    src.force_free = false;
    // move data
    src.data_ = nullptr;
    return *this;
}

// Begin iterator
array::Array::iterator array::Array::begin(void) {
    this->begin_ = intvec(this->ndim_, 0);
    this->end_ = intvec(this->ndim_, 0);
    this->end_[0] = this->shape_[0];
    return array::Array::iterator(this->begin_, this->shape_);
}

// End iterator
array::Array::iterator array::Array::end(void) {
    return array::Array::iterator(this->end_, this->shape_);
}

// Get value operator
float & array::Array::operator[] (const intvec & index) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<float *>(data_ptr));
}

// Get value of element at a n-dim index
float array::Array::get(const intvec & index) const {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<float *>(data_ptr));
}

// Get value of element at a C-contiguous index
float array::Array::get(std::uint64_t index) const {
    return this->get(contiguous_to_ndim_idx(index, this->shape()));
}

// Set value of element at a n-dim index
void array::Array::set(const intvec index, float value) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<float *>(data_ptr)) = value;
}

// Set value of element at a C-contiguous index
void array::Array::set(std::uint64_t index, float value) {
    this->set(contiguous_to_ndim_idx(index, this->shape()), value);
}

// Copy data from GPU array
#ifndef __MERLIN_CUDA__
void sync_from_gpu(const array::Parcel & gpu_array, const cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}
#endif  // __MERLIN_CUDA__

// Export data to a file
void array::Array::export_to_file(const std::string & filename) {
    array::Stock exported(filename, 'w');
    exported.get_metadata(*this);
    exported.write_metadata();
    exported.write_data_to_file(*this);
}

// Destructor
array::Array::~Array(void) {
    if (this->force_free) {
        free_memory(this->data_, this->size());
    }
}

}  // namespace merlin
