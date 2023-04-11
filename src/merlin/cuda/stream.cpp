// Copyright 2022 quocdang1998
#include "merlin/cuda/stream.hpp"

#include <sstream>  // std::ostringstream

#include "merlin/cuda/context.hpp"  // merlin::cuda::Context
#include "merlin/cuda/graph.hpp"    // merlin::cuda::Graph
#include "merlin/logger.hpp"        // cuda_compile_error, FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------------------------------------------------

// String representation
std::string cuda::Stream::str(void) const {
    std::ostringstream os;
    os << "<Stream at " << std::hex << this->stream_ << " associated to GPU " << this->device_.id() << ">";
    return os.str();
}

#ifndef __MERLIN_CUDA__

// Default constructor (the null stream)
cuda::Stream::Stream(void) {}

// Constructor from setting flag and priority
cuda::Stream::Stream(cuda::StreamSetting setting, int priority) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Get flag
cuda::StreamSetting cuda::Stream::setting(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return static_cast<cuda::StreamSetting>(0);
}

// Get priority
int cuda::Stream::priority(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return 0;
}

// Get context associated to stream
cuda::Context cuda::Stream::get_context(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return cuda::Context();
}

// Query for completion status
bool cuda::Stream::is_complete(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return false;
}

// Check if the stream is being captured
bool cuda::Stream::is_capturing(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return false;
}

// Check valid GPU and context
void cuda::Stream::check_cuda_context(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Add callback to stream
void cuda::Stream::add_callback(cuda::Stream::CudaStreamCallback func, void * arg) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Record event on a stream
void cuda::Stream::record_event(const cuda::Event & event) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Wait on an event
void cuda::Stream::wait_event(const cuda::Event & event, cuda::EventWaitFlag flag) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}
// Synchronize the stream
void cuda::Stream::synchronize(void) const {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// Destructor
cuda::Stream::~Stream(void) {}

// Capturing stream for CUDA graph
void cuda::begin_capture_stream(const cuda::Stream & stream, StreamCaptureMode mode) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
}

// End capturing a stream and returning a graph
cuda::Graph cuda::end_capture_stream(const cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for stream management.\n");
    return cuda::Graph();
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
