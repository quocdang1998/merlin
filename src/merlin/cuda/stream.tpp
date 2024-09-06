// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_STREAM_TPP_
#define MERLIN_CUDA_STREAM_TPP_

#include <tuple>        // std::apply, std::tuple
#include <type_traits>  // std::decay_t

namespace merlin {

#ifdef __NVCC__

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Wrapper callback around a function
template <typename Function, typename... Args>
void cuda::stream_callback_wrapper(::cudaStream_t stream, ::cudaError_t status, void * data) {
    std::uintptr_t * data_ptr = reinterpret_cast<std::uintptr_t *>(data);
    std::decay_t<Function> * p_callback = reinterpret_cast<std::decay_t<Function> *>(data_ptr[0]);
    std::tuple<Args...> * p_args = reinterpret_cast<std::tuple<Args...> *>(data_ptr[1]);
    std::apply(*p_callback, std::forward<std::tuple<Args...>>(*p_args));
    delete p_callback;
    delete p_args;
    delete[] data_ptr;
}

// ---------------------------------------------------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------------------------------------------------

// Append a CPU function to the stream
template <typename Function, typename... Args>
void cuda::Stream::add_callback(Function && callback, Args &&... args) {
    std::decay_t<Function> * p_callback = new std::decay_t<Function>(std::forward<Function>(callback));
    std::tuple<Args...> * p_args = new std::tuple<Args...>(std::forward<Args>(args)...);
    std::uintptr_t * data = new std::uintptr_t[2];
    data[0] = reinterpret_cast<std::uintptr_t>(p_callback);
    data[1] = reinterpret_cast<std::uintptr_t>(p_args);
    cuda::add_callback_to_stream(this->stream_, cuda::stream_callback_wrapper<Function, Args...>, data);
}

#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_CUDA_STREAM_TPP_
