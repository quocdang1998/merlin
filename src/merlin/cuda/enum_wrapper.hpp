// Copyright 2022 quocdang1998
#ifndef MERLIN_CUDA_ENUM_WRAPPER_HPP_
#define MERLIN_CUDA_ENUM_WRAPPER_HPP_

namespace merlin::cuda {

/** @brief Limit to get.*/
enum class DeviceLimit : unsigned int {
    /** @brief Size of the stack of each CUDA thread.*/
    StackSize = 0x00,
    /** @brief Size of the ``std::printf`` function buffer.*/
    PrintfSize = 0x01,
    /** @brief Size of the heap of each CUDA thread.*/
    HeapSize = 0x02,
    /** @brief Maximum nesting depth of a grid at which a thread can safely call ``cudaDeviceSynchronize``.*/
    SyncDepth = 0x03,
    /** @brief Maximum number of outstanding device runtime launches.*/
    LaunchPendingCount = 0x04
};

/** @brief Parameter describing the purpose of the event.*/
enum EventCategory : unsigned int {
    /** Default event.*/
    DefaultEvent = 0x00,
    /** Event meant to be synchronize with CPU (process on CPU blocked until the event occurs).*/
    BlockingSyncEvent = 0x01,
    /** Event not recording time data.*/
    DisableTimingEvent = 0x02,
    /** Event might be used in an interprocess communication.*/
    InterprocessEvent = 0x04
};

/** @brief Event wait flag.*/
enum class EventWaitFlag : unsigned int {
    /** Default event creation flag on stream.*/
    Default = 0x00,
    /** Event is captured in the graph as an external event node when performing stream capture.*/
    External = 0x01
};

/** @brief CUDA copy flag.*/
enum class MemcpyKind : unsigned int {
    /** @brief Copy flag from CPU memory to CPU memory.*/
    HostToHost = 0x0,
    /** @brief Copy flag from CPU memory to GPU memory.*/
    HostToDevice = 0x1,
    /** @brief Copy flag from GPU memory to CPU memory.*/
    DeviceToHost = 0x2,
    /** @brief Copy flag from GPU memory to GPU memory.*/
    DeviceToDevice = 0x3,
    /** @brief Auto detect transfer direction, requires unified virtual addressing.*/
    Default = 0x4
};

/** @brief CUDA graph node type.*/
enum class NodeType : unsigned int {
    /** @brief Kernel node.*/
    Kernel = 0x00,
    /** @brief Memory copy node.*/
    Memcpy = 0x01,
    /** @brief Memory set node.*/
    Memset = 0x02,
    /** @brief Host node.*/
    Host = 0x03,
    /** @brief Subgraph node.*/
    SubGraph = 0x04,
    /** @brief Empty node.*/
    Empty = 0x05,
    /** @brief Wait event node.*/
    WaitEvent = 0x06,
    /** @brief Record event node.*/
    RecordEvent = 0x07,
    /** @brief External semaphore signal node.*/
    ExtSemaphoreSignal = 0x08,
    /** @brief External semaphore wait node.*/
    ExtSemaphoreWait = 0x09,
    /** @brief Memory allocation node.*/
    MemAlloc = 0x0a,
    /** @brief Memory free node.*/
    MemFree = 0x0b
};

/** @brief Parameter controlling the behavior of the stream.*/
enum class StreamSetting : unsigned int {
    /** Default stream creation flag (synchonized with the null stream).*/
    Default = 0x00,
    /** Works may run concurrently with null stream.*/
    NonBlocking = 0x01
};

}  // namespace merlin::cuda

#endif  // MERLIN_CUDA_ENUM_WRAPPER_HPP_
