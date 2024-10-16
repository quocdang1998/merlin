// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_DECLARATION_HPP_
#define MERLIN_CUDA_DECLARATION_HPP_

namespace merlin::cuda {
class Device;    // GPU device
class CtxGuard;  // CUDA context lock guard
class Event;     // CUDA events (milestone in the stream)
class Stream;    // CUDA streams (queues of tasks)

struct GraphNode;  // CUDA graph node
class Graph;       // CUDA execution graph

template <typename... Args>
class Dispatcher;  // CUDA Memory copy interface
}  // namespace merlin::cuda

#endif  // MERLIN_CUDA_DECLARATION_HPP_
