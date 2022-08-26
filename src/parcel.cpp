// Copyright 2022 quocdang1998
#include "merlin/parcel.hpp"

#include "merlin/array.hpp"  // merlin::Array
#include "merlin/logger.hpp"  // FAILURE, cuda_compile_error

namespace merlin {

#ifndef MERLIN_CUDA_

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

// Free old data
void Parcel::free_current_data(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Destructor (do nothing)
Parcel::~Parcel(void) {}

#endif  // MERLIN_CUDA_

}  // namespace merlin
