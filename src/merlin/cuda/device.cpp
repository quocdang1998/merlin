// Copyright 2022 quocdang1998
#include "merlin/cuda/device.hpp"

#include "merlin/logger.hpp"  // merlin::Fatal, merlin::cuda_compile_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Construct a device from its ID
cuda::Device::Device(int id) {
    Fatal<cuda_compile_error>("Enable option MERLIN_CUDA at compilation to use this feature.\n");
}

// Get instance point to current GPU
cuda::Device cuda::Device::get_current_gpu(void) {
    Fatal<cuda_compile_error>("Enable option MERLIN_CUDA at compilation to use this feature.\n");
    return cuda::Device();
}

// Get total number of GPU
std::uint64_t cuda::Device::get_num_gpu(void) {
    Fatal<cuda_compile_error>("Enable option MERLIN_CUDA at compilation to use this feature.\n");
    return 0;
}

// Print limit of device
void cuda::Device::print_specification(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Test functionality of a GPU
bool cuda::Device::test_gpu(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return false;
}

// Set as current GPU
void cuda::Device::set_as_current(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Push the primary context associated to the GPU to the context stack
std::uintptr_t cuda::Device::push_context(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return 0;
}

// Pop the current context out of the context stack
void cuda::Device::pop_context(std::uintptr_t previous_context) noexcept {}

// Get and set limit
std::uint64_t cuda::Device::limit(cuda::DeviceLimit limit, std::uint64_t size) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return 0;
}

// Reset all GPU
void cuda::Device::reset_all(void) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Synchronize the current GPU
void cuda::Device::synchronize(void) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// String representation
std::string cuda::Device::str(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return "";
}

// Print limit of all GPU
void cuda::print_gpus_spec(void) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

// Test functionality of all GPU
bool cuda::test_all_gpu(void) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return false;
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
