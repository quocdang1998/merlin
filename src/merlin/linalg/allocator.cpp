// Copyright 2024 quocdang1998
#include "merlin/linalg/allocator.hpp"

#include "merlin/logger.hpp"    // merlin::Fatal
#include "merlin/platform.hpp"  // __MERLIN_WINDOWS__, __MERLIN_LINUX__

#if defined(__MERLIN_WINDOWS__)
    #include <malloc.h>  // ::_aligned_malloc, ::_aligned_free
#elif defined(__MERLIN_LINUX__)
    #include <cstdlib>  // std::aligned_alloc, std::free
#endif

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Aligned allocator
// ---------------------------------------------------------------------------------------------------------------------

// Align-alloc
double * linalg::aligned_alloc(std::size_t alignment, std::size_t size) {
#if defined(__MERLIN_WINDOWS__)
    return reinterpret_cast<double *>(::_aligned_malloc(size, alignment));
#elif defined(__MERLIN_LINUX__)
    return reinterpret_cast<double *>(std::aligned_alloc(alignment, size));
#endif
}

// Align-free
void linalg::aligned_free(double * ptr) {
#if defined(__MERLIN_WINDOWS__)
    ::_aligned_free(ptr);
#elif defined(__MERLIN_LINUX__)
    std::free(ptr);
#endif
}

}  // namespace merlin
