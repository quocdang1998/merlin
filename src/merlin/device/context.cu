// Copyright 2022 quocdang1998
#include "merlin/device/context.hpp"

#include "cuda.h"  // cuCtxCreate, cuCtxDestroy, CUcontext

#include "merlin/logger.hpp"  // cuda_get_error_name, cuda_runtime_error, FAILURE

namespace merlin::device {

Context::Context(void) {
    CUcontext ctx;
    CUresult err_ = cuCtxGetCurrent(&ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
    this->context_ = reinterpret_cast<std::uintptr_t>(ctx);
    this->device_ = Device();
    this->reference_ = true;
}

Context::Context(const Device & gpu, Context::Flags flag) {
    CUcontext ctx;
    CUresult err_ = cuCtxCreate(&ctx, static_cast<unsigned int>(flag), gpu.id());
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
    this->context_ = reinterpret_cast<std::uintptr_t>(ctx);
    this->device_ = gpu;
}

void Context::push_current(void) {
    if (this->attached_) {
        FAILURE(cuda_runtime_error, "The current context is being attached to another process\n");
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(this->context_);
    CUresult err_ = cuCtxPushCurrent(ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Push context to current failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
    this->attached_ = false;
}

Context & Context::pop_current(void) {
    if (!(this->attached_)) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to any process\n");
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(this->context_);
    CUresult err_ = cuCtxPopCurrent(&ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Pop current context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
    this->attached_ = true;
    return *this;
}

bool Context::is_current(void) {
    CUcontext current_ctx;
    CUresult err_ = cuCtxGetCurrent(&current_ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get current context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
    return (this->context_ == reinterpret_cast<std::uintptr_t>(current_ctx));
}

void Context::set_current(void) {
    if (!(this->attached_)) {
        FAILURE(cuda_runtime_error, "The current context is not being attached to any process\n");
    }
    CUcontext current_ctx = reinterpret_cast<CUcontext>(this->context_);
    CUresult err_ = cuCtxSetCurrent(current_ctx);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Set current context failed with message \"%s\".\n", cuda_get_error_name(err_));
    }
}

Context::~Context(void) {
    if ((this->context_ != NULL) && !(this->reference_)) {
        cuCtxDestroy(reinterpret_cast<CUcontext>(this->context_));
    }
}

}  // namespace merlin::device
