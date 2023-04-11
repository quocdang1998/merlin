// Copyright 2022 quocdang1998
#include "merlin/cuda/context.hpp"

#include <utility>  // std::pair

#include <cuda.h>  // ::cuCtxCreate, ::cuCtxDestroy, ::cuCtxGetCurrent, ::cuCtxGetDevice, ::cuCtxGetFlags,
                   // ::cuCtxPushCurrent, ::cuCtxPopCurrent, ::cuCtxSynchronize, ::CUcontext,
                   // ::cuDevicePrimaryCtxGetState, ::cuDevicePrimaryCtxRelease, ::cuDevicePrimaryCtxRetain

#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/logger.hpp"       // cuda_runtime_error, FAILURE

#define safety_lock() bool lock_success = Environment::mutex.try_lock()
#define safety_unlock()                                                                                                \
    if (lock_success) Environment::mutex.unlock()

namespace merlin {

// ----------------------------------------------------------------------------------------------------------------------
// CUDA Context
// ----------------------------------------------------------------------------------------------------------------------

// Get pointer to current context
static inline std::uintptr_t get_current_context_ptr(void) {
    // check for current context as regular context
    safety_lock();
    ::CUcontext current_ctx;
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxGetCurrent(&current_ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    // return if current context is a regular one
    if (current_ctx != nullptr) {
        return reinterpret_cast<std::uintptr_t>(current_ctx);
    }
    // if null pointer returned, query for primary context
    int num_gpu = cuda::Device::get_num_gpu();
    for (int i_gpu = 0; i_gpu < num_gpu; i_gpu++) {
        unsigned int flag;
        int active = 0;
        err_ = static_cast<::cudaError_t>(::cuDevicePrimaryCtxGetState(i_gpu, &flag, &active));
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "Query primary context for GPU ID %d failed with message \"%s\".\n", i_gpu,
                    ::cudaGetErrorString(err_));
        }
        if (active) {
            err_ = static_cast<::cudaError_t>(::cuDevicePrimaryCtxRetain(&current_ctx, i_gpu));
            if (err_ != 0) {
                FAILURE(cuda_runtime_error, "Get primary context for GPU ID %d failed with message \"%s\".\n", i_gpu,
                        ::cudaGetErrorString(err_));
            }
            std::uintptr_t context_ptr = reinterpret_cast<std::uintptr_t>(current_ctx);
            Environment::primary_contexts[i_gpu] = context_ptr;
            return context_ptr;
        }
    }
    // a dummy context initialized (return nullptr)
    safety_unlock();
    return reinterpret_cast<std::uintptr_t>(current_ctx);
}

// ----------------------------------------------------------------------------------------------------------------------
// Context
// ----------------------------------------------------------------------------------------------------------------------

// Member constructor
cuda::Context::Context(const cuda::Device & gpu, cuda::ContextSchedule schedule) {
    // Create context
    safety_lock();
    ::CUcontext ctx;
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxCreate(&ctx, static_cast<unsigned int>(schedule), gpu.id()));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create context failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    this->context_ = reinterpret_cast<std::uintptr_t>(ctx);
    // Increase reference count and initialize attached flag
    Environment::ContextAttribute attrib(1, cuda::Device::get_current_gpu().id());
    auto [it_current, success] = Environment::attribute.insert(std::pair(this->context_, attrib));
    if (!success) {
        FAILURE(cuda_runtime_error, "Create context failed because the context has already exist.\n");
    }
    safety_unlock();
}

// Constructor from context pointer (to be improved for the case of primary context)
cuda::Context::Context(std::uintptr_t context_ptr) : context_(context_ptr) {
    safety_lock();
    if (Environment::attribute.find(context_ptr) == Environment::attribute.end()) {
        Environment::attribute[context_ptr] = {0, cuda::Device::get_current_gpu().id()};
    }
    Environment::attribute[context_ptr].reference_count += 1;
    safety_unlock();
}

// Check if the context is the top of context stack
bool cuda::Context::is_current(void) const { return this->context_ == get_current_context_ptr(); }

// Push the context to the stack
void cuda::Context::push_current(void) const {
    safety_lock();
    ::CUcontext ctx = reinterpret_cast<::CUcontext>(this->context_);
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxPushCurrent(ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Push context to current stack failed with message \"%s\".\n",
                ::cudaGetErrorString(err_));
    }
    safety_unlock();
}

// Pop the context out of the stack
const cuda::Context & cuda::Context::pop_current(void) const {
    safety_lock();
    if (!(this->is_current())) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to the current processes.\n");
    }
    ::CUcontext ctx = reinterpret_cast<::CUcontext>(this->context_);
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxPopCurrent(&ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Pop current context out of the stack failed with message \"%s\".\n",
                ::cudaGetErrorString(err_));
    }
    safety_unlock();
    return *this;
}

// Get current context
cuda::Context cuda::Context::get_current(void) {
    safety_lock();
    cuda::Context result;
    result.context_ = get_current_context_ptr();
    if (Environment::attribute.find(result.context_) == Environment::attribute.end()) {
        int current_gpu = cuda::Device::get_current_gpu().id();
        Environment::attribute[result.context_] = Environment::ContextAttribute(1, current_gpu);
    }
    Environment::attribute[result.context_].reference_count += 1;
    safety_unlock();
    return result;
}

// Get GPU attached to current context
cuda::Device cuda::Context::get_gpu_of_current_context(void) {
    safety_lock();
    int device;
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxGetDevice(&device));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get GPU ID of current context failed with message \"%s\".\n",
                ::cudaGetErrorString(err_));
    }
    safety_unlock();
    return cuda::Device(device);
}

// Get flag of the current context.
cuda::ContextSchedule cuda::Context::get_flag_of_current_context(void) {
    safety_lock();
    unsigned int flag;
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxGetFlags(&flag));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get flag of current context failed with message \"%s\".\n",
                ::cudaGetErrorString(err_));
    }
    safety_unlock();
    return static_cast<cuda::ContextSchedule>(flag);
}

// Synchronize current context
void cuda::Context::synchronize(void) {
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuCtxSynchronize());
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Synchronize current context failed with message \"%s\".\n",
                ::cudaGetErrorString(err_));
    }
}

// Destructor
cuda::Context::~Context(void) {
    // free if the context is not a primary context and reference count goes to zero
    if (this->context_ != 0) {
        safety_lock();
        Environment::attribute[this->context_].reference_count -= 1;
        if (Environment::attribute[this->context_].reference_count == 0) {
            if (!this->is_primary()) {
                ::cuCtxDestroy(reinterpret_cast<::CUcontext>(this->context_));
            }
            Environment::attribute.erase(this->context_);
        }
        this->context_ = 0;
        safety_unlock();
    }
}

// Create a primary context attached to a GPU
cuda::Context cuda::create_primary_context(const cuda::Device & gpu, cuda::ContextSchedule flag) {
    // check validity of GPU
    if (gpu.id() < 0) {
        FAILURE(cuda_runtime_error, "Invalid GPU ID (id = %d).\n", gpu.id());
    }
    cuda::Context result;
    ::cudaError_t err_;
    // find already initialized context
    for (auto & [ctx_ptr, attribute] : Environment::attribute) {
        if (attribute.gpu == gpu.id()) {
            result.context_ = ctx_ptr;
            attribute.reference_count += 1;
            err_ = static_cast<::cudaError_t>(::cuDevicePrimaryCtxSetFlags(gpu.id(), static_cast<unsigned int>(flag)));
            if (err_ != 0) {
                FAILURE(cuda_runtime_error, "Set flag to primary context for GPU %d failed with message \"%s\".\n",
                        gpu.id(), ::cudaGetErrorName(err_));
            }
            return result;
        }
    }
    // retain context if not initialized
    ::CUcontext ctx;
    err_ = static_cast<::cudaError_t>(::cuDevicePrimaryCtxRetain(&ctx, gpu.id()));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create primary context for GPU %d failed with message \"%s\".\n", gpu.id(),
                ::cudaGetErrorName(err_));
    }
    result.context_ = reinterpret_cast<std::uintptr_t>(ctx);
    Environment::attribute.insert({
        result.context_, {1, gpu.id()}
    });
    err_ = static_cast<::cudaError_t>(::cuDevicePrimaryCtxSetFlags(gpu.id(), static_cast<unsigned int>(flag)));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Set flag to primary context for GPU %d failed with message \"%s\".\n", gpu.id(),
                ::cudaGetErrorName(err_));
    }
    return result;
}

}  // namespace merlin
