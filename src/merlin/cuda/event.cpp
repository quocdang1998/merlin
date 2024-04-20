// Copyright 2022 quocdang1998
#include "merlin/cuda/event.hpp"

#include <sstream>  // std::ostringstream

#include "merlin/logger.hpp"  // merlin::Fatal, merlin::cuda_compile_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------------------------------------------------

// String representation
std::string cuda::Event::str(void) const {
    std::ostringstream os;
    os << "<Event at " << std::hex << this->event_ << " associated to GPU " << this->device_.id() << ">";
    return os.str();
}

#ifndef __MERLIN_CUDA__

// Contruct an event with a given flag
cuda::Event::Event(unsigned int category) {}

// Query the status of works
bool cuda::Event::is_complete(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for event management.\n");
    return false;
}

// Check valid GPU and context
void cuda::Event::check_cuda_context(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for event management.\n");
}

// Synchronize the event
void cuda::Event::synchronize(void) const {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for event management.\n");
}

float cuda::operator-(const cuda::Event & ev_1, const cuda::Event & ev_2) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for event management.\n");
    return 0.0;
}

// Destructor
cuda::Event::~Event(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
