// Copyright 2023 quocdang1998
#include "merlin/candy/optimizer.hpp"

#include "merlin/logger.hpp"       // merlin::cuda_compile_error, FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__

// Create an object on GPU by the GPU
candy::Optimizer * candy::Optimizer::new_gpu(void) const {
    FAILURE(cuda_compile_error, "Compile with CUDA option enabled to access GPU features.\n");
    return nullptr;
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
