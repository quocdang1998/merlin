// Copyright 2022 quocdang1998
#include "merlin/cuda/stream.hpp"

#include <cuda.h>  // ::cuStreamGetCtx, ::CUcontext, ::CUstream

#include "merlin/cuda/context.hpp"  // merlin::cuda::Context
#include "merlin/cuda/event.hpp"  // merlin::cuda::Event
#include "merlin/logger.hpp"  // cuda_runtime_error, FAILURE, WARNING

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Stream
// --------------------------------------------------------------------------------------------------------------------

// Default constructor (the null stream)
cuda::Stream::Stream(void) {
    this->device_ = cuda::Device::get_current_gpu();
}

// Constructor from setting flag and priority
cuda::Stream::Stream(cuda::Stream::Setting setting, int priority) {
    // get min and max priority
    int min_priority, max_priority;
    ::cudaDeviceGetStreamPriorityRange(&min_priority, &max_priority);
    if ((priority > min_priority) || (priority < max_priority)) {
        WARNING("Priority out of range (expected priority in range [%d, %d], got %d), the priority will be clamped.",
                max_priority, min_priority, priority);
    }
    // create a stream within the context
    ::cudaStream_t stream;
    ::cudaError_t err_ = ::cudaStreamCreateWithPriority(&stream, static_cast<unsigned int>(setting), priority);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create stream failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    this->stream_ = reinterpret_cast<std::uintptr_t>(stream);
    this->device_ = cuda::Context::get_gpu_of_current_context();
}

// Get flag
cuda::Stream::Setting cuda::Stream::setting(void) const {
    unsigned int flag;
    ::cudaError_t err_ = ::cudaStreamGetFlags(reinterpret_cast<::cudaStream_t>(this->stream_), &flag);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get flag of stream failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    return static_cast<cuda::Stream::Setting>(flag);
}

// Get priority
int cuda::Stream::priority(void) const {
    int priority;
    ::cudaError_t err_ = ::cudaStreamGetPriority(reinterpret_cast<::cudaStream_t>(this->stream_), &priority);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Get priority of stream failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    return priority;
}

// Get context associated to stream
cuda::Context cuda::Stream::get_context(void) const {
    ::CUcontext context;
    ::CUstream stream = reinterpret_cast<::CUstream>(this->stream_);
    ::cudaError_t err_ = static_cast<::cudaError_t>(::cuStreamGetCtx(stream, &context));
    return cuda::Context(reinterpret_cast<std::uintptr_t>(context));
}

// Query for completion status
bool cuda::Stream::is_complete(void) const {
    ::cudaError_t err_ = ::cudaStreamQuery(reinterpret_cast<::cudaStream_t>(this->stream_));
    if (err_ == 0) {
        return true;
    } else if (err_ != 600) {
        FAILURE(cuda_runtime_error, "Query stream status failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    return false;
}

// Check valid GPU and context
void cuda::Stream::check_cuda_context(void) const {
    if (this->get_context() != cuda::Context::get_current()) {
        FAILURE(cuda_runtime_error, "Current context is not the one associated the stream.\n");
    }
    if (this->device_ != cuda::Context::get_gpu_of_current_context()) {
        FAILURE(cuda_runtime_error, "Current GPU is not the one associated the stream.\n");
    }
}

// Add callback to stream
void cuda::Stream::add_callback(cuda::CudaStreamCallback func, void * arg) {
    ::cudaError_t err_ = ::cudaStreamAddCallback(reinterpret_cast<::cudaStream_t>(this->stream_), func, arg, 0);
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Add callback to stream failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// Synchronize the stream
void cuda::Stream::synchronize(void) const {
    ::cudaError_t err_ = ::cudaStreamSynchronize(reinterpret_cast<::cudaStream_t>(this->stream_));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Stream synchronization failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// Destructor
cuda::Stream::~Stream(void) {
    if (this->stream_ != 0) {
        cudaError_t err_ = ::cudaStreamDestroy(reinterpret_cast<cudaStream_t>(this->stream_));
        if (err_ != cudaSuccess) {
            FAILURE(cuda_runtime_error, "cudaStreamDestroy failed with message \"%s\".\n", ::cudaGetErrorName(err_));
        }
    }
}

// Record event on a stream
void cuda::record_event(const cuda::Event & event, const cuda::Stream & stream) {
    stream.check_cuda_context();
    event.check_cuda_context();
    ::cudaError_t err_ = ::cudaEventRecord(reinterpret_cast<::cudaEvent_t>(event.get_event_ptr()),
                                           reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Record event failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

}  // namespace merlin
