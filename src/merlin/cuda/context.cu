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
    // Increase reference count and initialize attached flag
    Context::m_.lock();
    auto [it_current, success] = Context::shared_attributes_.insert({this->context_, {1, true, gpu}});
    if (!success) {
        FAILURE(cuda_runtime_error, "Create context failed because the context has already exist.\n");
    }
    Context::m_.unlock();
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
    Context::m_.lock();
    Context::shared_attributes_[this->context_].attached = true;
    Context::m_.unlock();
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
    Context::m_.lock();
    Context::shared_attributes_[this->context_].attached = false;
    Context::m_.unlock();
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
    Context::m_.lock();
    if (Context::shared_attributes_.find(result.context_) == Context::shared_attributes_.end()) {
        Context::shared_attributes_[result.context_] = {1, true, Device::get_current_gpu()};
    }
    Context::shared_attributes_[result.context_].reference_count += 1;
    Context::m_.unlock();
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

// Destructor
Context::~Context(void) {
    // free if the context is not a primary context and reference count goes to zero
    if (this->context_ != 0) {
        Context::m_.lock();
        if (--Context::shared_attributes_[this->context_].reference_count == 0) {
            Context::shared_attributes_.erase(this->context_);
            cuCtxDestroy(reinterpret_cast<CUcontext>(this->context_));
        }
        Context::m_.unlock();
    }
}

// --------------------------------------------------------------------------------------------------------------------
// PrimaryContext
// --------------------------------------------------------------------------------------------------------------------

}  // namespace merlin::cuda
