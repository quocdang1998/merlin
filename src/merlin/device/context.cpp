// Copyright 2022 quocdang1998
#include "merlin/device/context.hpp"

namespace merlin::device {

// -------------------------------------------------------------------------------------------------------------------------
// Context
// -------------------------------------------------------------------------------------------------------------------------

// List of primary contexts
std::vector<Context> Context::primary_contexts;

#ifndef __MERLIN_CUDA__

// Member constructor
Context::Context(const Device & gpu, Context::Flags flag) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Push the context to the stack
void Context::push_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Pop the context out of the stack
Context & Context::pop_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return *this;
}

// Check if the context is the top of context stack
bool Context::is_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return false;
}

// Set current context at the top of the stack
void Context::set_current(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// List of primary contexts
std::vector<Context> Context::primary_contexts;

// Create list of primary contexts
void Context::create_primary_context_list(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Create primary context instance assigned to a GPU
Context Context::create_primary_context(const Device & gpu) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return Context();
}

// Get state of the primary context
std::pair<bool, Context::Flags> Context::get_primary_ctx_state(const Device & gpu) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
    return std::pair<bool, Context::Flags>(false, Context::Flags::AutoSchedule);
}

// Set flag for primary context
void Context::set_flag_primary_context(const Device & gpu, Context::Flags flag) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

// Destructor
Context::~Context(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for context management.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin::device
