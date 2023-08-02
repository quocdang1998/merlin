// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <cstdio>      // std::fread, std::fseek
#include <cstring>     // std::memcpy
#include <functional>  // std::bind, std::placeholders
#include <ios>         // std::ios_base::failure
#include <mutex>       // std::mutex
#include <utility>     // std::move

#include "merlin/array/copy.hpp"    // merlin::array::contiguous_strides, merlin::array::array_copy
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/array/slice.hpp"   // merlin::array::Slice
#include "merlin/array/stock.hpp"   // merlin::array::Stock
#include "merlin/logger.hpp"        // FAILURE
#include "merlin/utils.hpp"         // merlin::contiguous_to_ndim_idx, merlin::inner_prod

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Memory lock (allocated array always stays in the RAM)
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Allocate non pageable memory
double * array::allocate_memory(std::uint64_t size) {
    double * result = new double[size];
    if (result == nullptr) {
        FAILURE(std::runtime_error, "Cannot allocate memory.\n");
    }
    return result;
}

// Pin memory to RAM
void array::cuda_pin_memory(double * ptr, std::uint64_t n_elem) {}

// Free non pageable memory
void array::free_memory(double * ptr) { delete[] ptr; }

#endif  // __MERLIN_CUDA__

// ---------------------------------------------------------------------------------------------------------------------
// Read data
// ---------------------------------------------------------------------------------------------------------------------

// Read an array from file
static inline void read_from_file(double * dest, std::FILE * file, double * src, std::uint64_t bytes) {
    std::fseek(file, reinterpret_cast<std::uintptr_t>(src), SEEK_SET);
    std::uint64_t count = bytes / sizeof(double);
    if (std::fread(dest, sizeof(double), count, file) != count) {
        FAILURE(std::ios_base::failure, "Read file error.\n");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------------------------------------------------

// Initialize begin and end iterator
void array::Array::initialize_iterator(void) noexcept {
    if (this->ndim() == 0) {
        return;
    }
    intvec index(this->ndim(), 0);
    this->begin_ = array::Array::iterator(index, this->shape_, this->data_);
    index[0] = this->shape_[0];
    this->end_ = array::Array::iterator(index, this->shape_, this->data_);
}

// Constructor Array of one element
array::Array::Array(double value) {
    // allocate data
    this->data_ = array::allocate_memory(1);
    this->data_[0] = value;

    // set metadata
    this->size_ = 1;
    this->strides_ = intvec({sizeof(double)});
    this->shape_ = intvec({1});
    this->release_ = true;
    this->initialize_iterator();
}

// Construct empty Array from shape vector
array::Array::Array(const intvec & shape) : array::NdData(shape) {
    // initialize data
    this->data_ = array::allocate_memory(this->size());
    // other meta data
    this->release_ = true;
    this->initialize_iterator();
}

// Construct Array from Numpy array
array::Array::Array(double * data, const intvec & shape, const intvec & strides, bool copy) {
    this->shape_ = shape;
    this->calc_array_size();
    this->release_ = copy;
    // copy / assign data
    if (copy) {  // copy data
        // allocate a new tensor
        this->data_ = array::allocate_memory(this->size());
        // reform the stride tensor (force into C shape)
        this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
        // copy data from old tensor to new tensor (optimized with memcpy)
        array::NdData src(data, shape, strides);
        array::array_copy(dynamic_cast<array::NdData *>(this), &src, std::memcpy);
    } else {
        this->strides_ = strides;
        this->data_ = data;
        array::cuda_pin_memory(this->data_, this->size());
    }
    this->initialize_iterator();
}

// Constructor from a slice
array::Array::Array(const array::Array & whole, const Vector<array::Slice> & slices) : array::NdData(whole, slices) {
    this->release_ = false;
    this->initialize_iterator();
}

// Copy constructor
array::Array::Array(const array::Array & src) : array::NdData(src) {
    // copy / initialize meta data
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    this->release_ = true;
    // copy data
    this->data_ = array::allocate_memory(this->size());
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), std::memcpy);
    this->initialize_iterator();
}

// Copy assignment
array::Array & array::Array::operator=(const array::Array & src) {
    // copy / initialize meta data
    this->array::NdData::operator=(src);
    this->strides_ = array::contiguous_strides(this->shape_, sizeof(double));
    // free current data
    if (this->release_) {
        delete[] this->data_;
    }
    this->release_ = true;
    // copy data
    this->data_ = array::allocate_memory(this->size());
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), std::memcpy);
    this->initialize_iterator();
    return *this;
}

// Move constructor
array::Array::Array(array::Array && src) : array::NdData(src) {
    // disable release_ of the source
    this->release_ = src.release_;
    src.release_ = false;
    // nullify source data
    src.data_ = nullptr;
    this->initialize_iterator();
}

// Move assignment
array::Array & array::Array::operator=(array::Array && src) {
    // disable release_ of the source and free current data
    if (this->release_) {
        array::free_memory(this->data_);
    }
    // copy meta data
    this->array::NdData::operator=(src);
    this->release_ = src.release_;
    src.release_ = false;
    // move data
    src.data_ = nullptr;
    this->initialize_iterator();
    return *this;
}

// Get value operator
double & array::Array::operator[](const intvec & index) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get value operator
const double & array::Array::operator[](const intvec & index) const {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<const double *>(data_ptr));
}

// Get value of element at a n-dim index
double array::Array::get(const intvec & index) const {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    return *(reinterpret_cast<double *>(data_ptr));
}

// Get value of element at a C-contiguous index
double array::Array::get(std::uint64_t index) const { return this->get(contiguous_to_ndim_idx(index, this->shape())); }

// Set value of element at a n-dim index
void array::Array::set(const intvec index, double value) {
    std::uint64_t leap = inner_prod(index, this->strides_);
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(this->data_) + leap;
    *(reinterpret_cast<double *>(data_ptr)) = value;
}

// Set value of element at a C-contiguous index
void array::Array::set(std::uint64_t index, double value) {
    this->set(contiguous_to_ndim_idx(index, this->shape()), value);
}

// Reshape
void array::Array::reshape(const intvec & new_shape) {
    this->array::NdData::reshape(new_shape);
    this->initialize_iterator();
}

// Collapse dimension from felt (or right)
void array::Array::remove_dim(std::uint64_t i_dim) {
    this->array::NdData::remove_dim(i_dim);
    this->initialize_iterator();
}

// Copy data from GPU array
#ifndef __MERLIN_CUDA__
void array::Array::clone_data_from_gpu(const array::Parcel & src, const cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}
#endif  // __MERLIN_CUDA__

// Export data to a file
void array::Array::extract_data_from_file(const array::Stock & src) {
    auto read_func = std::bind(read_from_file, std::placeholders::_1, src.get_file_ptr(), std::placeholders::_2,
                               std::placeholders::_3);
    if (src.is_thread_safe()) {
        src.get_file_lock().lock();
    }
    array::array_copy(this, &src, read_func);
    if (src.is_thread_safe()) {
        src.get_file_lock().unlock();
    }
}

// Destructor
array::Array::~Array(void) {
    if (this->release_ && (this->data_ != nullptr)) {
        array::free_memory(this->data_);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Shuffle array
// ---------------------------------------------------------------------------------------------------------------------

// Shuffle array
array::Array array::shuffle_array(const array::Array & src, const Shuffle & suffle_index) {
    array::Array result(src.shape());
    std::uint64_t size = src.size();
    for (std::uint64_t i_data = 0; i_data < size; i_data++) {
        intvec index_data = contiguous_to_ndim_idx(i_data, src.shape());
        intvec new_index_data = suffle_index[index_data];
        result[new_index_data] = src.get(index_data);
    }
    return result;
}

/** @brief Read an array with shuffled index.*/
array::Array array::shuffled_read(const array::Stock & src, const Shuffle & suffle_index) {
    array::Array result(src.shape());
    std::uint64_t size = src.size();
    for (std::uint64_t i_data = 0; i_data < size; i_data++) {
        intvec index_data = contiguous_to_ndim_idx(i_data, src.shape());
        intvec new_index_data = suffle_index[index_data];
        result[new_index_data] = src.get(index_data);
    }
    return result;
}

}  // namespace merlin
