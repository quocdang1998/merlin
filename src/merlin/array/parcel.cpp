// Copyright 2022 quocdang1998
#include "merlin/array/parcel.hpp"

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"  // FAILURE, cuda_compile_error

namespace merlin::array {

// -------------------------------------------------------------------------------------------------------------------------
// Parcel
// -------------------------------------------------------------------------------------------------------------------------

// Initialize mutex
std::mutex Parcel::m_;

#ifndef __MERLIN_CUDA__

// Default constructor
Parcel::Parcel(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Constructor from CPU array
Parcel::Parcel(const Array & cpu_array, std::uintptr_t stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Constructor from a slice
Parcel::Parcel(const Parcel & whole, std::initializer_list<Slice> slices) {
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

// Copy data to a pre-allocated memory
void Parcel::copy_to_gpu(Parcel * gpu_ptr, void * shape_strides_ptr) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Free old data
void Parcel::free_current_data(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to access Parcel feature.\n");
}

// Destructor (do nothing)
Parcel::~Parcel(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin::array
