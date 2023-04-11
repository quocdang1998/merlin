// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_DECLARATION_HPP_
#define MERLIN_CUDA_DECLARATION_HPP_

namespace merlin::cuda {
class Device;   // GPU device
class Context;  // CUDA runtime context
class Event;    // CUDA events (milestone in the stream)
class Stream;   // CUDA streams (queues of tasks)
class Graph;    // CUDA Execution Graph
}  // namespace merlin::cuda

#endif  // MERLIN_CUDA_DECLARATION_HPP_
