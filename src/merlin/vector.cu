// Copyright 2024 quocdang1998
#include "merlin/vector.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Vector
// ---------------------------------------------------------------------------------------------------------------------

// CUDA copy functions (CPU to GPU)
void vector_cpy_to_gpu(void * dest, const void * src, std::uint64_t size, std::uintptr_t stream_ptr) {
    ::cudaMemcpyAsync(dest, src, size, ::cudaMemcpyHostToDevice, reinterpret_cast<::cudaStream_t>(stream_ptr));
}

// CUDA copy functions (GPU to CPU)
void vector_cpy_from_gpu(void * dest, const void * src, std::uint64_t size, std::uintptr_t stream_ptr) {
    ::cudaMemcpyAsync(dest, src, size, ::cudaMemcpyDeviceToHost, reinterpret_cast<::cudaStream_t>(stream_ptr));
}

}  // namespace merlin
