// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_GRAPH_TPP_
#define MERLIN_CUDA_GRAPH_TPP_

#include "merlin/logger.hpp"  // merlin::Fatal, merlin::cuda_compile_error, merlin::cuda_runtime_error

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------------------------------------------------

#ifndef __MERLIN_CUDA__
// Add CUDA kernel node
template <typename Function, typename... Args>
cuda::GraphNode cuda::Graph::add_kernel_node(Function * kernel, std::uint64_t n_blocks, std::uint64_t n_threads,
                                             std::uint64_t shared_mem, const Vector<cuda::GraphNode> & deps,
                                             Args &&... args) {
    Fatal<cuda_compile_error>("Compile merlin with CUDA by enabling option MERLIN_CUDA for graph management.\n");
    return cuda::GraphNode();
}
#elif defined(__NVCC__)
// Add CUDA kernel node
template <typename Function, typename... Args>
cuda::GraphNode cuda::Graph::add_kernel_node(Function * kernel, std::uint64_t n_blocks, std::uint64_t n_threads,
                                             std::uint64_t shared_mem, const Vector<cuda::GraphNode> & deps,
                                             Args &&... args) {
    // initialize kernel param
    ::cudaKernelNodeParams kernel_param;
    kernel_param.func = reinterpret_cast<void *>(kernel);
    kernel_param.gridDim = n_blocks;
    kernel_param.blockDim = n_threads;
    kernel_param.sharedMemBytes = shared_mem;
    std::array<void *, sizeof...(args)> kernel_args = {{reinterpret_cast<void *>(&args)...}};
    kernel_param.kernelParams = kernel_args.data();
    kernel_param.extra = nullptr;
    // add node to graph
    ::cudaGraphNode_t graph_node;
    Vector<::cudaGraphNode_t> dependancies(deps.size());
    for (std::uint64_t i = 0; i < dependancies.size(); i++) {
        dependancies[i] = reinterpret_cast<::cudaGraphNode_t>(deps[i].graphnode_ptr);
    }
    ::cudaError_t err_ = ::cudaGraphAddKernelNode(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_ptr_),
                                                  dependancies.data(), dependancies.size(), &kernel_param);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add kernel node to graph failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}
#endif  // __MERLIN_CUDA__

}  // namespace merlin

#endif  // MERLIN_CUDA_GRAPH_TPP_
