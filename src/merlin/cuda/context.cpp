// Copyright 2022 quocdang1998
#include "merlin/cuda/context.hpp"

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------------------------------------------------

// Mutex lock for updating static attributes
std::mutex cuda::Context::m_;

// Attributes of instances
std::map<std::uintptr_t, cuda::Context::SharedAttribures> cuda::Context::shared_attributes;

#ifndef __MERLIN_CUDA__

// Member constructor
cuda::Context::Context(const cuda::Device & gpu, cuda::Context::Flags flag) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Increase reference count
void cuda::Context::increase_reference_count(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Decrease reference count
void cuda::Context::decrease_reference_count(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
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

// Check if the context is the top of context stack
bool cuda::Context::is_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return false;
}

// Set current context at the top of the stack
void cuda::Context::set_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// String representation
std::string cuda::Context::repr(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
    return "";
}

// Destructor
cuda::Context::~Context(void) {}

#endif  // __MERLIN_CUDA__

// --------------------------------------------------------------------------------------------------------------------
// PrimaryContext
// --------------------------------------------------------------------------------------------------------------------

}  // namespace merlin::cuda
