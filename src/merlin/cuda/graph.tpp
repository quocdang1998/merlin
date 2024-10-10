// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_GRAPH_TPP_
#define MERLIN_CUDA_GRAPH_TPP_

#include <tuple>        // std::apply, std::tuple
#include <type_traits>  // std::decay_t

#include "merlin/logger.hpp"  // merlin::cuda_runtime_error, merlin::Fatal

namespace merlin {

#ifdef __NVCC__

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Wrapper callback around a host function for graph
template <typename Function, typename... Args>
void cuda::graph_callback_wrapper(void * data) {
    std::uintptr_t * data_ptr = reinterpret_cast<std::uintptr_t *>(data);
    std::decay_t<Function> * p_callback = reinterpret_cast<std::decay_t<Function> *>(data_ptr[0]);
    std::tuple<Args...> * p_args = reinterpret_cast<std::tuple<Args...> *>(data_ptr[1]);
    std::apply(*p_callback, std::forward<std::tuple<Args...>>(*p_args));
    delete p_callback;
    delete p_args;
    delete[] data_ptr;
}

// ---------------------------------------------------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------------------------------------------------

// Add CUDA kernel node
template <typename Function, typename... Args>
cuda::GraphNode cuda::Graph::add_kernel_node(Function * kernel, std::uint64_t n_blocks, std::uint64_t n_threads,
                                             std::uint64_t shared_mem, const cuda::GraphNodeList & deps,
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
    ::cudaError_t err_ = ::cudaGraphAddKernelNode(&graph_node, reinterpret_cast<::cudaGraph_t>(this->graph_),
                                                  reinterpret_cast<const ::cudaGraphNode_t *>(deps.data()), deps.size(),
                                                  &kernel_param);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Add kernel node to graph failed with message \"%s\".\n", ::cudaGetErrorString(err_));
    }
    return cuda::GraphNode(reinterpret_cast<std::uintptr_t>(graph_node));
}

// Add CUDA host node
template <typename Function, typename... Args>
cuda::GraphNode cuda::Graph::add_host_node(Function && callback, const cuda::GraphNodeList & deps, Args &&... args) {
    std::decay_t<Function> * p_callback = new std::decay_t<Function>(std::forward<Function>(callback));
    std::tuple<Args...> * p_args = new std::tuple<Args...>(std::forward<Args>(args)...);
    std::uintptr_t * data = new std::uintptr_t[2];
    data[0] = reinterpret_cast<std::uintptr_t>(p_callback);
    data[1] = reinterpret_cast<std::uintptr_t>(p_args);
    return cuda::add_callback_to_graph(this->graph_, cuda::graph_callback_wrapper<Function, Args...>, deps, data);
}

#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_CUDA_GRAPH_TPP_
