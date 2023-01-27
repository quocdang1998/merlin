// Copyright 2022 quocdang1998
#include "merlin/cuda/context.hpp"

#include <sstream>  // std::ostringstream

#include "merlin/cuda/device.hpp"  // merlin::cuda::Device

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------------------------------------------------

// Mutex lock for updating static attributes
std::mutex cuda::Context::mutex_;

// Attributes of instances
std::map<std::uintptr_t, cuda::Context::Attribute> cuda::Context::attribute_;

// String representation
std::string cuda::Context::str(void) {
    std::ostringstream os;
    os << "<Context instance at " << std::hex << this->context_ << std::dec
       << ", reference count " << this->get_reference_count() << ">";
    return os.str();
}

// Default context
cuda::Context default_context = cuda::initialize_context();

#ifndef __MERLIN_CUDA__

// Member constructor
cuda::Context::Context(const cuda::Device & gpu, cuda::Context::Flags flag) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Check if the context is the top of context stack
bool cuda::Context::is_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return false;
}

// Push the context to the stack
void cuda::Context::push_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Pop the context out of the stack
cuda::Context & cuda::Context::pop_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return *this;
}

// Get current context
cuda::Context cuda::Context::get_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return Context();
}

// Get GPU attached to current context
cuda::Device cuda::Context::get_gpu_of_current_context(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return cuda::Device();
}

// Get flag of the current context.
cuda::Context::Flags cuda::Context::get_flag_of_current_context(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return cuda::Context::Flags::AutoSchedule;
}

// Synchronize current context
void cuda::Context::synchronize(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Destructor
cuda::Context::~Context(void) {}

// Create a primary context attached to a GPU
cuda::Context cuda::create_primary_context(const cuda::Device & gpu, cuda::Context::Flags flag) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return cuda::Context();
}

// Initialize a default context
cuda::Context cuda::initialize_context(void) {
    return cuda::Context();
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
