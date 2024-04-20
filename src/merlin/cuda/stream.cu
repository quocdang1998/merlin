// Copyright 2022 quocdang1998
#include "merlin/cuda/stream.hpp"

#include <algorithm>  // std::clamp

#include "merlin/cuda/event.hpp"  // merlin::cuda::Event
#include "merlin/cuda/graph.hpp"  // merlin::cuda::Graph
#include "merlin/logger.hpp"      // merlin::Fatal, merlin::Warning, merlin::cuda_runtime_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------------------------------------------------

// Wrapper of the function adding CUDA callback to stream
void cuda::cuda_stream_add_callback(std::uintptr_t stream, cuda::StreamCallback func, void * arg) {
    ::cudaError_t err_ = ::cudaStreamAddCallback(reinterpret_cast<::cudaStream_t>(stream), func, arg, 0);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add callback to stream failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------------------------------------------------

// Default constructor (the null stream)
cuda::Stream::Stream(void) { this->device_ = cuda::Device::get_current_gpu(); }

// Constructor from setting flag and priority
cuda::Stream::Stream(cuda::StreamSetting setting, int priority) {
    // get min and max priority
    int min_priority, max_priority;
    ::cudaDeviceGetStreamPriorityRange(&min_priority, &max_priority);
    if ((priority > min_priority) || (priority < max_priority)) {
        Warning("Priority out of range (expected priority in range [%d, %d], got %d), the priority will be clamped.",
                max_priority, min_priority, priority);
    }
    priority = std::clamp(priority, min_priority, max_priority);
    // create a stream within the context
    ::cudaStream_t stream;
    ::cudaError_t err_ = ::cudaStreamCreateWithPriority(&stream, static_cast<unsigned int>(setting), priority);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Create stream failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    this->stream_ = reinterpret_cast<std::uintptr_t>(stream);
    this->device_ = cuda::Device::get_current_gpu();
}

// Get flag
cuda::StreamSetting cuda::Stream::get_setting(void) const {
    unsigned int flag;
    ::cudaError_t err_ = ::cudaStreamGetFlags(reinterpret_cast<::cudaStream_t>(this->stream_), &flag);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Get flag of stream failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    return static_cast<cuda::StreamSetting>(flag);
}

// Get priority
int cuda::Stream::get_priority(void) const {
    int priority;
    ::cudaError_t err_ = ::cudaStreamGetPriority(reinterpret_cast<::cudaStream_t>(this->stream_), &priority);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Get priority of stream failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    return priority;
}

// Query for completion status
bool cuda::Stream::is_complete(void) const {
    ::cudaError_t err_ = ::cudaStreamQuery(reinterpret_cast<::cudaStream_t>(this->stream_));
    if (err_ == 0) {
        return true;
    } else if (err_ != 600) {
        Fatal<cuda_runtime_error>("Query stream status failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    return false;
}

// Check if the stream is being captured
bool cuda::Stream::is_capturing(void) const {
    ::cudaStreamCaptureStatus capture_status;
    ::cudaError_t err_ = ::cudaStreamIsCapturing(reinterpret_cast<::cudaStream_t>(this->stream_), &capture_status);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Query stream capture status failed with message \"%s\".\n",
                ::cudaGetErrorString(err_));
    }
    if (capture_status == 2) {
        Warning("Stream is capturing, but end capture is not beeing called.\n");
    }
    if (capture_status > 0) {
        return true;
    }
    return false;
}

// Check valid GPU and context
void cuda::Stream::check_cuda_context(void) const {
    if ((this->device_ != cuda::Device::get_current_gpu()) && (this->stream_ != 0)) {
        Fatal<cuda_runtime_error>("Current GPU is not the one associated the stream.\n");
    }
}

// Record event on a stream
void cuda::Stream::record_event(const cuda::Event & event) const {
    this->check_cuda_context();
    event.check_cuda_context();
    ::cudaError_t err_ = ::cudaEventRecord(reinterpret_cast<::cudaEvent_t>(event.get_event_ptr()),
                                           reinterpret_cast<::cudaStream_t>(this->stream_));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Record event failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
}

// Wait on an event
void cuda::Stream::wait_event(const cuda::Event & event, cuda::EventWaitFlag flag) const {
    ::cudaError_t err_ = ::cudaStreamWaitEvent(reinterpret_cast<::cudaStream_t>(this->stream_),
                                               reinterpret_cast<::cudaEvent_t>(event.get_event_ptr()),
                                               static_cast<unsigned int>(flag));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Record event failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
}

// Synchronize the stream
void cuda::Stream::synchronize(void) const {
    ::cudaError_t err_ = ::cudaStreamSynchronize(reinterpret_cast<::cudaStream_t>(this->stream_));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Stream synchronization failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
}

// Destructor
cuda::Stream::~Stream(void) {
    if (this->stream_ != 0) {
        cudaError_t err_ = ::cudaStreamDestroy(reinterpret_cast<cudaStream_t>(this->stream_));
        if (err_ != 0) {
            Fatal<cuda_runtime_error>("cudaStreamDestroy failed with message \"%s\".\n", ::cudaGetErrorString(err_));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Stream capturing
// ---------------------------------------------------------------------------------------------------------------------

// Capturing stream for CUDA graph
void cuda::begin_capture_stream(const cuda::Stream & stream, StreamCaptureMode mode) {
    if (stream.is_capturing()) {
        Fatal<cuda_runtime_error>("Cannot re-capture a capturing stream.\n");
    }
    ::cudaError_t err_ = ::cudaStreamBeginCapture(reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()),
                                                  static_cast<::cudaStreamCaptureMode>(mode));
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Capture stream failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
}

// End capturing a stream and returning a graph
cuda::Graph cuda::end_capture_stream(const cuda::Stream & stream) {
    ::cudaGraph_t graph_ptr;
    ::cudaError_t err_ = ::cudaStreamEndCapture(reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr()), &graph_ptr);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Retunr graph from captured stream failed with message \"%s\".\n",
                ::cudaGetErrorString(err_));
    }
    return cuda::Graph(reinterpret_cast<std::uintptr_t>(graph_ptr));
}

}  // namespace merlin
