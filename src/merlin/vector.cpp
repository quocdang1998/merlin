// Copyright 2024 quocdang1998
#include "merlin/vector.hpp"

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Vector
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// CUDA copy functions (CPU to GPU)
void vector_cpy_to_gpu(void * dest, const void * src, std::uint64_t size, std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Compile the library with CUDA option to enable data transfering to GPU.\n");
}

// CUDA copy functions (GPU to CPU)
void vector_cpy_from_gpu(void * dest, const void * src, std::uint64_t size, std::uintptr_t stream_ptr) {
    Fatal<cuda_compile_error>("Compile the library with CUDA option to enable data transfering from GPU.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
