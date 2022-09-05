// Copyright 2022 quocdang1998
#include "merlin/parcel.hpp"

#include "merlin/array.hpp"  // merlin::Array
#include "merlin/logger.hpp"  // FAILURE, cuda_compile_error

namespace merlin {

#ifndef __MERLIN_CUDA__

// Default constructor
Parcel::Parcel(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Constructor from CPU array
Parcel::Parcel(const Array & cpu_array, uintptr_t stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Copy constructor
Parcel::Parcel(const Parcel & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Copy assignment
Parcel & Parcel::operator=(const Parcel & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
    return *this;
}

// Move constructor
Parcel::Parcel(Parcel && src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Move assignment
Parcel & Parcel::operator=(Parcel && src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
    return *this;
}

// Check if current device holds data pointed by object
int Parcel::check_device(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Copy data to a pre-allocated memory
void Parcel::copy_to_gpu(Parcel * gpu_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Free old data
void Parcel::free_current_data(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Destructor (do nothing)
Parcel::~Parcel(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
