// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"  // FAILURE, cuda_compile_error

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Parcel
// --------------------------------------------------------------------------------------------------------------------

// Initialize mutex
std::mutex array::Parcel::mutex_;

#ifndef __MERLIN_CUDA__

// Copy constructor
array::Parcel::Parcel(const array::Parcel & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Copy assignment
array::Parcel & array::Parcel::operator=(const array::Parcel & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
    return *this;
}

// Move constructor
array::Parcel::Parcel(array::Parcel && src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Move assignment
array::Parcel & array::Parcel::operator=(array::Parcel && src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
    return *this;
}

// Copy data to a pre-allocated memory
void * array::Parcel::copy_to_gpu(array::Parcel * gpu_ptr, void * shape_strides_ptr) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
    return nullptr;
}

// Free old data
void array::Parcel::free_current_data(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Destructor (do nothing)
array::Parcel::~Parcel(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
