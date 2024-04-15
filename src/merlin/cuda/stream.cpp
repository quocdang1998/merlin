// Copyright 2022 quocdang1998
#include "merlin/cuda/stream.hpp"

#include <sstream>  // std::ostringstream

#include "merlin/cuda/graph.hpp"  // merlin::cuda::Graph
#include "merlin/logger.hpp"      // merlin::Fatal, merlin::cuda_compile_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------------------------------------------------

// String representation
std::string cuda::Stream::str(void) const {
    std::ostringstream os;
    if (this->stream_ == 0) {
        os << "<Default stream>";
    } else {
        os << "<Stream at " << std::hex << this->stream_ << " associated to GPU " << this->device_.id() << ">";
    }
    return os.str();
}

#ifndef __MERLIN_CUDA__

// Default constructor (the null stream)
cuda::Stream::Stream(void) {}

// Constructor from setting flag and priority
cuda::Stream::Stream(cuda::StreamSetting setting, int priority) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Get flag
cuda::StreamSetting cuda::Stream::get_setting(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return static_cast<cuda::StreamSetting>(0);
}

// Get priority
int cuda::Stream::get_priority(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return 0;
}

// Query for completion status
bool cuda::Stream::is_complete(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return false;
}

// Check if the stream is being captured
bool cuda::Stream::is_capturing(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return false;
}

// Check valid GPU and context
void cuda::Stream::check_cuda_context(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Add callback to stream
void cuda::Stream::add_callback(cuda::Stream::CudaStreamCallback func, void * arg) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Record event on a stream
void cuda::Stream::record_event(const cuda::Event & event) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Wait on an event
void cuda::Stream::wait_event(const cuda::Event & event, cuda::EventWaitFlag flag) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}
// Synchronize the stream
void cuda::Stream::synchronize(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Destructor
cuda::Stream::~Stream(void) {}

// Capturing stream for CUDA graph
void cuda::begin_capture_stream(const cuda::Stream & stream, StreamCaptureMode mode) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// End capturing a stream and returning a graph
cuda::Graph cuda::end_capture_stream(const cuda::Stream & stream) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return cuda::Graph();
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
