// Copyright 2022 quocdang1998
#include "merlin/device/gpu_query.hpp"

#include "merlin/logger.hpp"  // FAILURE, cuda_compile_error

namespace merlin::device {

#ifndef __MERLIN_CUDA__

// Print limit of device
void Device::print_specification(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Test functionality of a GPU
bool Device::test_gpu(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return false;
}

// Set as current GPU
void Device::set_as_current(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Get and set limit
std::uint64_t Device::limit(Device::Limit limit, std::uint64_t size) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return 0;
}

// Reset all GPU
void Device::reset_all(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// String representation
std::string Device::repr(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return "";
}

// Print limit of all GPU
void print_all_gpu_specification(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Test functionality of all GPU
bool test_all_gpu(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return false;
}

// Map from GPU ID to is details
std::map<int, Device> gpu_map;

#endif  // __MERLIN_CUDA__

}  // namespace merlin::device
