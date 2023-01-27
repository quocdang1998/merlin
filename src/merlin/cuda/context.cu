// Copyright 2022 quocdang1998
#include "merlin/cuda/context.hpp"

#include <tuple>  // std::get, std::tuple

#include <cuda.h>  // ::cuCtxCreate, ::cuCtxDestroy, ::cuCtxGetCurrent, ::cuCtxGetDevice, ::cuCtxGetFlags,
                   // ::cuCtxPushCurrent, ::cuCtxPopCurrent, ::cuCtxSynchronize, ::CUcontext,
                   // ::cuDevicePrimaryCtxGetState, ::cuDevicePrimaryCtxRelease, ::cuDevicePrimaryCtxRetain

#include "merlin/cuda/device.hpp"  // merlin::cuda::Device

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CUDA Context
// --------------------------------------------------------------------------------------------------------------------

// Get pointer to current context
static inline std::tuple<std::uintptr_t, bool, int> get_current_context_ptr(void) {
    // check for current context as regular context
    ::CUcontext current_ctx;
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxGetCurrent(&current_ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // return if current context is a regular one
    if (current_ctx != nullptr) {
        return {reinterpret_cast<std::uintptr_t>(current_ctx), false, -1};
    }
    // if null pointer returned, query for primary context
    int num_gpu = cuda::Device::get_num_gpu();
    for (int i_gpu = 0; i_gpu < num_gpu; i_gpu++) {
        unsigned int flag;
        int active = 0;
        err_ = static_cast<cudaError_t>(::cuDevicePrimaryCtxGetState(i_gpu, &flag, &active));
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "Query primary context for GPU ID %d failed with message \"%s\".\n", i_gpu,
                    ::cudaGetErrorName(err_));
        }
        if (active) {
            err_ = static_cast<cudaError_t>(::cuDevicePrimaryCtxRetain(&current_ctx, i_gpu));
            if (err_ != 0) {
                FAILURE(cuda_runtime_error, "Get primary context for GPU ID %d failed with message \"%s\".\n", i_gpu,
                        ::cudaGetErrorName(err_));
            }
            return {reinterpret_cast<std::uintptr_t>(current_ctx), true, i_gpu};
        }
    }
    // a dummy context initialized (return nullptr)
    return {reinterpret_cast<std::uintptr_t>(current_ctx), false, -1};
}

// --------------------------------------------------------------------------------------------------------------------
// Context
// --------------------------------------------------------------------------------------------------------------------

// Member constructor
cuda::Context::Context(const cuda::Device & gpu, cuda::Context::Flags flag) {
    // Create context
    ::CUcontext ctx;
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxCreate(&ctx, static_cast<unsigned int>(flag), gpu.id()));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create context failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    this->context_ = reinterpret_cast<std::uintptr_t>(ctx);
    // Increase reference count and initialize attached flag
    cuda::Context::mutex_.lock();
    auto [it_current, success] = cuda::Context::attribute_.insert({this->context_, {1, false, -1}});
    if (!success) {
        FAILURE(cuda_runtime_error, "Create context failed because the context has already exist.\n");
    }
    cuda::Context::mutex_.unlock();
}

// Constructor from context pointer (to be improved for the case of primary context)
cuda::Context::Context(std::uintptr_t context_ptr) : context_(context_ptr) {
    cuda::Context::mutex_.lock();
    if (cuda::Context::attribute_.find(context_ptr) == cuda::Context::attribute_.end()) {
        cuda::Context::attribute_[context_ptr] = {0, false, -1};
    }
    cuda::Context::attribute_[context_ptr].reference_count += 1;
    cuda::Context::mutex_.unlock();
}

// Check if the context is the top of context stack
bool cuda::Context::is_current(void) {
    return this->context_ == std::get<0>(get_current_context_ptr());
}

// Push the context to the stack
void cuda::Context::push_current(void) {
    if (this->is_current()) {
        FAILURE(cuda_runtime_error, "The current context is being attached to the CPU process\n");
    }
    ::CUcontext ctx = reinterpret_cast<::CUcontext>(this->context_);
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxPushCurrent(ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Push context to current stack failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
}

// Pop the context out of the stack
cuda::Context & cuda::Context::pop_current(void) {
    if (!(this->is_current())) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to any processes\n");
    }
    ::CUcontext ctx = reinterpret_cast<::CUcontext>(this->context_);
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxPopCurrent(&ctx));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Pop current context out of the stack failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
    return *this;
}

// Get current context
cuda::Context cuda::Context::get_current(void) {
    cuda::Context result;
    auto [p_ctx, primarity, gpu_id] = get_current_context_ptr();
    result.context_ = p_ctx;
    cuda::Context::mutex_.lock();
    if (cuda::Context::attribute_.find(result.context_) == cuda::Context::attribute_.end()) {
        cuda::Context::attribute_[result.context_].reference_count = 1;
        if (primarity) {
            cuda::Context::attribute_[result.context_].is_primary = true;
            cuda::Context::attribute_[result.context_].gpu = gpu_id;
        }
    }
    cuda::Context::attribute_[result.context_].reference_count += 1;
    cuda::Context::mutex_.unlock();
    return result;
}

// Get GPU attached to current context
cuda::Device cuda::Context::get_gpu_of_current_context(void) {
    int device;
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxGetDevice(&device));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get GPU ID of current context failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
    return cuda::Device(device);
}

// Get flag of the current context.
cuda::Context::Flags cuda::Context::get_flag_of_current_context(void) {
    unsigned int flag;
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxGetFlags(&flag));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get flag of current context failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
    return static_cast<cuda::Context::Flags>(flag);
}

// Synchronize current context
void cuda::Context::synchronize(void) {
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxSynchronize());
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Synchronize current context failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
}

// Destructor
cuda::Context::~Context(void) {
    // free if the context is not a primary context and reference count goes to zero
    if (this->context_ != 0) {
        cuda::Context::mutex_.lock();
        cuda::Context::attribute_[this->context_].reference_count -= 1;
        cuda::Context::mutex_.unlock();
        if (cuda::Context::attribute_[this->context_].reference_count == 0) {
            if (this->is_primary()) {
                ::cuDevicePrimaryCtxRelease(cuda::Context::attribute_[this->context_].gpu);
            } else {
                ::cuCtxDestroy(reinterpret_cast<CUcontext>(this->context_));
            }
            cuda::Context::attribute_.erase(this->context_);
        }
    }
}

// Create a primary context attached to a GPU
cuda::Context cuda::create_primary_context(const cuda::Device & gpu, cuda::Context::Flags flag) {
    // check validity of GPU
    if (gpu.id() < 0) {
        FAILURE(cuda_runtime_error, "Invalid GPU ID (id = %d).\n", gpu.id());
    }
    cuda::Context result;
    cudaError_t err_;
    // find already initialized context
    cuda::Context::mutex_.lock();
    for (auto & [ctx_ptr, attribute] : cuda::Context::attribute_) {
        if (attribute.gpu == gpu.id()) {
            result.context_ = ctx_ptr;
            attribute.reference_count += 1;
            cuda::Context::mutex_.unlock();
            err_ = static_cast<cudaError_t>(::cuDevicePrimaryCtxSetFlags(gpu.id(), static_cast<unsigned int>(flag)));
            if (err_ != 0) {
                FAILURE(cuda_runtime_error, "Set flag to primary context for GPU %d failed with message \"%s\".\n",
                        gpu.id(), ::cudaGetErrorName(err_));
            }
            return result;
        }
    }
    // retain context if not initialized
    ::CUcontext ctx;
    err_ = static_cast<cudaError_t>(::cuDevicePrimaryCtxRetain(&ctx, gpu.id()));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create primary context for GPU %d failed with message \"%s\".\n", gpu.id(),
                ::cudaGetErrorName(err_));
    }
    result.context_ = reinterpret_cast<std::uintptr_t>(ctx);
    cuda::Context::attribute_.insert({result.context_, {1, true, gpu.id()}});
    cuda::Context::mutex_.unlock();
    err_ = static_cast<cudaError_t>(::cuDevicePrimaryCtxSetFlags(gpu.id(), static_cast<unsigned int>(flag)));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Set flag to primary context for GPU %d failed with message \"%s\".\n", gpu.id(),
                ::cudaGetErrorName(err_));
    }
    return result;
}

// Initialize a default context
cuda::Context cuda::initialize_context(void) {
    // check for number of GPU
    int num_gpu = cuda::Device::get_num_gpu();
    if (num_gpu == 0) {
        // return empty context if no GPU was found
        WARNING("No GPU was found. Return empty CUDA context (GPU functions will have no effect).\n");
        return cuda::Context();
    }
    // check for current context as regular context
    ::CUcontext current_ctx;
    cudaError_t err_ = static_cast<cudaError_t>(::cuCtxGetCurrent(&current_ctx));
    if ((err_ != 3) && (err_ != 0)) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    // uninitilized case, or initialized case but result is nullptr
    if ((err_ == 3) || (current_ctx == nullptr)) {
        return cuda::Context(cuda::Device::get_current_gpu(), cuda::Context::Flags::AutoSchedule);
    }
    // initialized case (return success)
    cuda::Context result;
    result.context_ = reinterpret_cast<std::uint64_t>(current_ctx);
    cuda::Context::mutex_.lock();
    if (cuda::Context::attribute_.find(result.context_) == cuda::Context::attribute_.end()) {
        cuda::Context::attribute_[result.context_] = {1, false, -1};
    }
    cuda::Context::attribute_[result.context_].reference_count += 1;
    cuda::Context::mutex_.unlock();
    return result;
}

}  // namespace merlin
