// Copyright 2022 quocdang1998
#include "merlin/cuda/device.hpp"

#include "merlin/logger.hpp"  // FAILURE, cuda_compile_error

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Device
// --------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Print limit of device
void cuda::Device::print_specification(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Test functionality of a GPU
bool cuda::Device::test_gpu(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return false;
}

// Set as current GPU
void cuda::Device::set_as_current(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Get and set limit
std::uint64_t cuda::Device::limit(cuda::DeviceLimit limit, std::uint64_t size) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return 0;
}

// Reset all GPU
void cuda::Device::reset_all(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// String representation
std::string cuda::Device::str(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return "";
}

// Print limit of all GPU
void cuda::print_all_gpu_specification(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Test functionality of all GPU
bool cuda::test_all_gpu(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return false;
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
