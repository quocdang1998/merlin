// Copyright 2022 quocdang1998
#include "merlin/cuda/context.hpp"

#include <sstream>  // std::ostringstream

#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/env.hpp"          // merlin::Environment
#include "merlin/logger.hpp"       // cuda_runtime_error, FAILURE

namespace merlin {

// ----------------------------------------------------------------------------------------------------------------------
// Context
// ----------------------------------------------------------------------------------------------------------------------

// Check if context is primary
bool cuda::Context::is_primary(void) const {
    for (const auto & [i_gpu, primary_context] : Environment::primary_contexts) {
        if (this->context_ == primary_context) {
            return true;
        }
    }
    return false;
}

// String representation
std::string cuda::Context::str(void) {
    std::ostringstream os;
    os << "<Context instance at " << std::hex << this->context_ << std::dec << ", reference count "
       << this->get_reference_count() << ">";
    return os.str();
}

#ifndef __MERLIN_CUDA__

// Member constructor
cuda::Context::Context(const cuda::Device & gpu, cuda::ContextSchedule schedule) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Constructor from context pointer (to be improved for the case of primary context)
cuda::Context::Context(std::uintptr_t context_ptr) : context_(context_ptr) { }

// Check if the context is the top of context stack
bool cuda::Context::is_current(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return false;
}

// Push the context to the stack
void cuda::Context::push_current(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Pop the context out of the stack
const cuda::Context & cuda::Context::pop_current(void) const {
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
cuda::ContextSchedule cuda::Context::get_flag_of_current_context(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return cuda::ContextSchedule::Auto;
}

// Synchronize current context
void cuda::Context::synchronize(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Destructor
cuda::Context::~Context(void) {}

// Create a primary context attached to a GPU
cuda::Context cuda::create_primary_context(const cuda::Device & gpu, cuda::ContextSchedule flag) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return cuda::Context();
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
