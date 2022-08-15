// Copyright 2022 quocdang1998
#include "merlin/parcel.hpp"

#include "merlin/tensor.hpp"  // Tensor
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

#ifndef MERLIN_CUDA_

// Default constructor
Parcel::Parcel(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Constructor from CPU array
Parcel::Parcel(const Tensor & cpu_array, uintptr_t stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Copy constructor
Parcel::Parcel(const Parcel & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Copy assignment
Parcel & Parcel::operator=(const Parcel & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Move constructor
Parcel::Parcel(Parcel && src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Free old data
void Parcel::free_current_data(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Update the shape vector and strides vector on GPU memory
void Parcel::copy_metadata(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Destructor (do nothing)
Parcel::~Parcel(void) {}

#endif  // MERLIN_CUDA_

}  // namespace merlin
