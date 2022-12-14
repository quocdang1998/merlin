// Copyright 2022 quocdang1998
#include "merlin/cuda/event.hpp"

#include "merlin/logger.hpp"  // cuda_runtime_error, FAILURE, WARNING

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Event
// --------------------------------------------------------------------------------------------------------------------

// Contruct an event with a given flag
cuda::Event::Event(cuda::Event::Category category) : category_(category), device_(cuda::Device::get_current_gpu()),
context_(cuda::Context::get_current()) {
    ::cudaEvent_t event;
    ::cudaError_t err_ = ::cudaEventCreateWithFlags(&event, static_cast<unsigned int>(category));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Create event failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    this->event_ = reinterpret_cast<std::uintptr_t>(event);
}

// Query the status of works
bool cuda::Event::is_complete(void) const {
    ::cudaError_t err_ = ::cudaEventQuery(reinterpret_cast<::cudaEvent_t>(this->event_));
    if ((err_ != 0) && (err_ != 600)) {
        FAILURE(cuda_runtime_error, "Query event failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
    if (err_ == 0) {
        return true;
    }
    return false;
}

// Check valid GPU and context
void cuda::Event::check_cuda_context(void) const {
    if (this->context_ != cuda::Context::get_current()) {
        FAILURE(cuda_runtime_error, "Current context is not the one associated the event.\n");
    }
    if (this->device_ != cuda::Context::get_gpu_of_current_context()) {
        FAILURE(cuda_runtime_error, "Current GPU is not the one associated the event.\n");
    }
}

// Synchronize the event
void cuda::Event::synchronize(void) const {
    ::cudaError_t err_ = ::cudaEventSynchronize(reinterpret_cast<::cudaEvent_t>(this->event_));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Event synchronization failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// Get elapsed time between 2 events
float cuda::operator-(const cuda::Event & ev_1, const cuda::Event & ev_2) {
    float result;
    ::cudaError_t err_ = ::cudaEventElapsedTime(&result, reinterpret_cast<::cudaEvent_t>(ev_1.event_),
                                                reinterpret_cast<::cudaEvent_t>(ev_2.event_));
    if (err_ != 0) {
        FAILURE(cuda_runtime_error, "Calculate elapsed time between 2 event failed with message \"%s\".\n",
                ::cudaGetErrorName(err_));
    }
    return result;
}

// Destructor
cuda::Event::~Event(void) {
    if (this->event_ != 0) {
        ::cudaError_t err_ = ::cudaEventDestroy(reinterpret_cast<::cudaEvent_t>(this->event_));
        if (err_ != 0) {
            FAILURE(cuda_runtime_error, "Destroy event failed with message \"%s\".\n", ::cudaGetErrorName(err_));
        }
    }
}

}  // namespace merlin
