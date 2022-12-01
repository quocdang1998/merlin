// Copyright 2022 quocdang1998
#include "merlin/cuda/stream.hpp"

#include "merlin/logger.hpp"  // cuda_runtime_error, FAILURE, WARNING

namespace merlin {

// Constructor from setting flag and priority
cuda::Stream::Stream(cuda::Context & context, cuda::Stream::Setting setting, int priority) {
    // get min and max priority
    int min_priority, max_priority;
    ::cudaDeviceGetStreamPriorityRange(&min_priority, &max_priority);
    if ((priority > min_priority) || (priority < max_priority)) {
        WARNING("Priority out of range (expected priority in range [%d, %d], got %d), the priority will be clamped.",
                max_priority, min_priority, priority);
    }
    // create a stream within the context
    cudaStream_t stream;
    bool is_current = true;
    if (!context.is_current()) {
        is_current = false;
        context.push_current();
    }
    cudaError_t err_ = ::cudaStreamCreateWithPriority(&stream, static_cast<unsigned int>(setting), priority);
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaStreamCreate failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    this->stream_ = reinterpret_cast<std::uintptr_t>(stream);
    this->context_ = &context;
    if (!is_current) {
        context.pop_current();
    }
}

// Get flag
cuda::Stream::Setting cuda::Stream::setting(void) {
    unsigned int flag;
    cudaError_t err_ = ::cudaStreamGetFlags(reinterpret_cast<cudaStream_t>(this->stream_), &flag);
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaStreamGetFlags failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    return static_cast<cuda::Stream::Setting>(flag);
}

// Get priority
int cuda::Stream::priority(void) {
    int priority;
    cudaError_t err_ = ::cudaStreamGetPriority(reinterpret_cast<cudaStream_t>(this->stream_), &priority);
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaStreamGetPriority failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    return priority;
}

// Query for completion status
bool cuda::Stream::is_complete(void) {
    cudaError_t err_ = ::cudaStreamQuery(reinterpret_cast<cudaStream_t>(this->stream_));
    if (err_ == cudaSuccess) {
        return true;
    } else if (err_ != cudaErrorNotReady) {
        FAILURE(cuda_runtime_error, "cudaStreamQuery failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    return false;
}

// Add callback to stream
void cuda::Stream::launch_cpu_function(CudaStreamCallback func, void * arg) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(this->stream_), reinterpret_cast<cudaHostFn_t>(func), arg);
}

// Synchronize the stream
void cuda::Stream::synchronize(void) {
    cudaError_t err_ = ::cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(this->stream_));
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaStreamSynchronize failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// Destructor
cuda::Stream::~Stream(void) {
    cudaError_t err_ = ::cudaStreamDestroy(reinterpret_cast<cudaStream_t>(this->stream_));
    if (err_ != cudaSuccess) {
        FAILURE(cuda_runtime_error, "cudaStreamDestroy failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

}  // namespace merlin
