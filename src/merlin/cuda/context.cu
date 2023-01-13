// Copyright 2022 quocdang1998
#include "merlin/cuda/context.hpp"

#include <sstream>  // std::ostringstream

#include "cuda.h"  // cuCtxCreate, cuCtxDestroy, CUcontext

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------------------------------------------------

// Member constructor
cuda::Context::Context(const cuda::Device & gpu, cuda::Context::Flags flag) {
    // Create context
    CUcontext ctx;
    cudaError_t err_ = static_cast<cudaError_t>(cuCtxCreate(&ctx, static_cast<unsigned int>(flag), gpu.id()));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create context failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    this->context_ = reinterpret_cast<std::uintptr_t>(ctx);
    // Increase reference count and initialize attached flag
    cuda::Context::m_.lock();
    auto [it_current, success] = cuda::Context::shared_attributes.insert({this->context_, {1, true, gpu}});
    if (!success) {
        FAILURE(cuda_runtime_error, "Create context failed because the context has already exist.\n");
    }
    cuda::Context::m_.unlock();
}

// Increase reference count
void cuda::Context::increase_reference_count(void) {
    cuda::Context::shared_attributes[this->context_].reference_count += 1;
}

// Decrease reference count
void cuda::Context::decrease_reference_count(void) {
    cuda::Context::shared_attributes[this->context_].reference_count -= 1;
}

// Push the context to the stack
void cuda::Context::push_current(void) {
    if (this->is_attached()) {
        FAILURE(cuda_runtime_error, "The current context is being attached to the CPU process\n");
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(this->context_);
    cudaError_t err_ = static_cast<cudaError_t>(cuCtxPushCurrent(ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Push context to current stack failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
    cuda::Context::m_.lock();
    cuda::Context::shared_attributes[this->context_].attached = true;
    cuda::Context::m_.unlock();
}

// Pop the context out of the stack
cuda::Context & cuda::Context::pop_current(void) {
    if (!(this->is_attached())) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to any processes\n");
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(this->context_);
    cudaError_t err_ = static_cast<cudaError_t>(cuCtxPopCurrent(&ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Pop current context out of the stack failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
    cuda::Context::m_.lock();
    cuda::Context::shared_attributes[this->context_].attached = false;
    cuda::Context::m_.unlock();
    return *this;
}

// Get current context
cuda::Context cuda::Context::get_current(void) {
    Context result;
    CUcontext current_ctx;
    cudaError_t err_ = static_cast<cudaError_t>(cuCtxGetCurrent(&current_ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    result.context_ = reinterpret_cast<std::uintptr_t>(current_ctx);
    cuda::Context::m_.lock();
    if (cuda::Context::shared_attributes.find(result.context_) == cuda::Context::shared_attributes.end()) {
        cuda::Context::shared_attributes[result.context_] = {1, true, cuda::Device::get_current_gpu()};
    }
    cuda::Context::shared_attributes[result.context_].reference_count += 1;
    cuda::Context::m_.unlock();
    return result;
}

// Check if the context is the top of context stack
bool cuda::Context::is_current(void) {
    cuda::Context current = cuda::Context::get_current();
    return (this->context_ == current.context_);
}

// Set current context at the top of the stack
void cuda::Context::set_current(void) {
    if (!(this->is_attached())) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to any process\n");
    }
    CUcontext current_ctx = reinterpret_cast<CUcontext>(this->context_);
    cudaError_t err_ = static_cast<cudaError_t>(cuCtxSetCurrent(current_ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Set current context failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// String representation
std::string cuda::Context::repr(void) {
    std::ostringstream os;
    os << "<Context instance at " << std::hex << this->context_ << std::dec << ">";
    return os.str();
}

// Destructor
cuda::Context::~Context(void) {
    // free if the context is not a primary context and reference count goes to zero
    if (this->context_ != 0) {
        cuda::Context::m_.lock();
        if (--cuda::Context::shared_attributes[this->context_].reference_count == 0) {
            cuda::Context::shared_attributes.erase(this->context_);
            cuCtxDestroy(reinterpret_cast<CUcontext>(this->context_));
        }
        cuda::Context::m_.unlock();
    }
}

// --------------------------------------------------------------------------------------------------------------------
// PrimaryContext
// --------------------------------------------------------------------------------------------------------------------

}  // namespace merlin
