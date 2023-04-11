// Copyright 2022 quocdang1998
#include "merlin/cuda/graph.hpp"

#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/logger.hpp"  // cuda_compile_error, FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Graph
// --------------------------------------------------------------------------------------------------------------------

// Mutex
std::mutex & cuda::Graph::mutex_ = Environment::mutex;

#ifndef __MERLIN_CUDA__

// Destroy current CUDA graph instance
void cuda::Graph::destroy_graph(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Constructor
cuda::Graph::Graph(int flag) {}

// Copy constructor
cuda::Graph::Graph(const cuda::Graph & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Copy assignment
cuda::Graph & cuda::Graph::operator=(const cuda::Graph & src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Move constructor
cuda::Graph::Graph(cuda::Graph && src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Move assignment
cuda::Graph & cuda::Graph::operator=(cuda::Graph && src) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Execute a graph
void cuda::Graph::execute(const cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
}

// Destructor
cuda::Graph::~Graph(void) {}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
