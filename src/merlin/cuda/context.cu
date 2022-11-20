// Copyright 2022 quocdang1998
#include "merlin/cuda/context.hpp"

#include "cuda.h"  // cuCtxCreate, cuCtxDestroy, CUcontext

namespace merlin::cuda {

// --------------------------------------------------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------------------------------------------------

// Member constructor
Context::Context(const Device & gpu, Context::Flags flag) {
    // Create context
    CUcontext ctx;
    CUresult err_ = cuCtxCreate(&ctx, static_cast<unsigned int>(flag), gpu.id());
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
    this->context_ = reinterpret_cast<std::uintptr_t>(ctx);
    this->device_ = gpu;
    // Increase reference count and initialize attached flag
    auto [it_current, success] = Context::reference_count_.insert({this->context_, 1});
    if (!success) {
        FAILURE(cuda_runtime_error, "Create context failed because the context has already exist.\n");
    }
    Context::attached_[this->context_] = true;
}

// Push the context to the stack
void Context::push_current(void) {
    if (this->is_attached()) {
        FAILURE(cuda_runtime_error, "The current context is being attached to the CPU process\n");
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(this->context_);
    CUresult err_ = cuCtxPushCurrent(ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Push context to current stack failed with message \"%s\".\n",
                cuda_get_error_name(err_));
    }
    Context::attached_[this->context_] = true;
}

// Pop the context out of the stack
Context & Context::pop_current(void) {
    if (!(this->is_attached())) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to any processes\n");
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(this->context_);
    CUresult err_ = cuCtxPopCurrent(&ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Pop current context out of the stack failed with message \"%s\".\n",
                cuda_get_error_name(err_));
    }
    Context::attached_[this->context_] = false;
    return *this;
}

// Get current context
Context Context::get_current(void) {
    Context result;
    CUcontext current_ctx;
    CUresult err_ = cuCtxGetCurrent(&current_ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
    result.context_ = reinterpret_cast<std::uintptr_t>(current_ctx);
    result.device_ = Device::get_current_gpu();
    if (Context::reference_count_.find(result.context_) == Context::reference_count_.end()) {
        Context::reference_count_[result.context_] = 1;
    }
    Context::reference_count_[result.context_] += 1;
    return result;
}

// Check if the context is the top of context stack
bool Context::is_current(void) {
    Context current = Context::get_current();
    return (this->context_ == current.context_);
}

// Set current context at the top of the stack
void Context::set_current(void) {
    if (!(this->is_attached())) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to any process\n");
    }
    CUcontext current_ctx = reinterpret_cast<CUcontext>(this->context_);
    CUresult err_ = cuCtxSetCurrent(current_ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Set current context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
}

// Create list of primary contexts
void Context::create_primary_context_list(void) {
    int num_gpu = Device::get_num_gpu();
    // skip when the primary contexts are initialized
    if (num_gpu == Context::primary_contexts.size()) {
        return;
    }
    for (int i = 0; i < num_gpu; i++) {
        Context::primary_contexts.emplace_back(Context::create_primary_context(Device(i)));
    }
}

// Create primary context instance assigned to a GPU
Context Context::create_primary_context(const Device & gpu) {
    Context result;
    result.device_ = gpu;
    auto [active, _] = result.get_primary_ctx_state(gpu);
    Context::reference_count_[result.context_] = 1;
    Context::attached_[result.context_] = active;
    return result;
}

// Get state of the primary context
std::pair<bool, Context::Flags> Context::get_primary_ctx_state(const Device & gpu) {
    unsigned int flags;
    int active;
    CUresult err_ = cuDevicePrimaryCtxGetState(gpu.id(), &flags, &active);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get state of primary context of GPU %d failed with message \"%s\".\n",
                gpu.id(), cuda_get_error_name(err_));
    }
    return std::pair<bool, Context::Flags>(static_cast<bool>(active), Context::Flags(flags));
}

// Set flag for primary context
void Context::set_flag_primary_context(const Device & gpu, Context::Flags flag) {
    CUresult err_ = cuDevicePrimaryCtxSetFlags(gpu.id(), static_cast<unsigned int>(flag));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Set flag to primary context of GPU %d failed with message \"%s\".\n",
                gpu.id(), cuda_get_error_name(err_));
    }
}

// Destructor
Context::~Context(void) {
    // free if the context is not a primary context and reference count goes to zero
    if (this->context_ != 0) {
        Context::reference_count_[this->context_] -= 1;
        if (Context::reference_count_[this->context_] == 0) {
            Context::reference_count_.erase(this->context_);
            Context::attached_.erase(this->context_);
            cuCtxDestroy(reinterpret_cast<CUcontext>(this->context_));
        }
    }
}

}  // namespace merlin::cuda
