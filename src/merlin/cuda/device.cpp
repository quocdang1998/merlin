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
    Fatal<cuda_compile_error>("Cannot detect GPU {} in non-CUDA mode.\n", id);
}

// Get instance point to current GPU
cuda::Device cuda::Device::get_current_gpu(void) {
    Fatal<cuda_compile_error>("Cannot detect the current GPU in non-CUDA mode.\n");
    return cuda::Device();
}

// Get total number of GPU
std::uint64_t cuda::Device::get_num_gpu(void) {
    Fatal<cuda_compile_error>("Cannot query the total number of GPUs in non-CUDA mode.\n");
    return 0;
}

// Print limit of device
void cuda::Device::print_specification(void) const {
    Fatal<cuda_compile_error>("Cannot get GPU specifications in non-CUDA mode.\n");
}

// Test functionality of a GPU
bool cuda::Device::test_gpu(void) const {
    Fatal<cuda_compile_error>("Cannot check for functionality of GPU in non-CUDA mode.\n");
    return false;
}

// Set as current GPU
void cuda::Device::set_as_current(void) const {
    Fatal<cuda_compile_error>("Cannot change current GPU in non-CUDA mode.\n");
}

// Push the primary context associated to the GPU to the context stack
std::uintptr_t cuda::Device::push_context(void) const {
    Fatal<cuda_compile_error>("Cannot push context associated to the requested GPU in non-CUDA mode.\n");
    return 0;
}

// Pop the current context out of the context stack
void cuda::Device::pop_context(std::uintptr_t previous_context) noexcept {}

// Get and set limit
std::uint64_t cuda::Device::limit(cuda::DeviceLimit limit, std::uint64_t size) {
    Fatal<cuda_compile_error>("Cannot query and modify GPU limits in non-CUDA mode.\n");
    return 0;
}

// Reset all GPU
void cuda::Device::reset_all(void) {
    Fatal<cuda_compile_error>("Cannot reset GPU state in non-CUDA mode.\n");
}

// Synchronize the current GPU
void cuda::Device::synchronize(void) {
    Fatal<cuda_compile_error>("Cannot synchronize the current GPU in non-CUDA mode.\n");
}

// String representation
std::string cuda::Device::str(void) const {
    Fatal<cuda_compile_error>("Cannot detect GPU in non-CUDA mode.\n");
    return "";
}

// Print limit of all GPU
void cuda::print_gpus_spec(void) {
    Fatal<cuda_compile_error>("Cannot get GPU specifications in non-CUDA mode.\n");
}

// Test functionality of all GPU
bool cuda::test_all_gpu(void) {
    Fatal<cuda_compile_error>("Cannot check for functionality of GPU in non-CUDA mode.\n");
    return false;
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
