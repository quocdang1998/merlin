# Copyright 2022 quocdang1998

cdef extern from "merlin/cuda/device.hpp":

    cpdef enum class DeviceLimit "merlin::cuda::Device::Limit":
        """GPU limit flags.

        *Values*

         - ``StackSize``: Size of the stack of each CUDA thread.
         - ``PrintfSize``: Size of the ``std::printf`` function buffer.
         - ``HeapSize``: Size of the heap of each CUDA thread.
         - ``SyncDepth``: Maximum nesting depth of a grid at which a thread can safely call ``cudaDeviceSynchronize``.
         - ``LaunchPendingCount``: Maximum number of outstanding device runtime launches.
        """
        StackSize,
        PrintfSize,
        HeapSize,
        SyncDepth,
        LaunchPendingCount

cdef extern from "merlin/cuda/context.hpp":

    cpdef enum class ContextFlags "merlin::cuda::Context::Flags":
        """
        CUDA Context setting flags.

        *Values*

         - ``AutoSchedule``: Automatic schedule based on the number of context and number of logical process.
         - ``SpinSchedule``: Actively spins when waiting for results from the GPU.
         - ``YieldSchedule``: Yield the CPU process when waiting for results from the GPU.
         - ``BlockSyncSchedule``: Block CPU process until synchronization.
        """
        AutoSchedule,
        SpinSchedule,
        YieldSchedule,
        BlockSyncSchedule

cdef extern from "merlin/cuda/event.hpp":

    cpdef enum class EventCategory "merlin::cuda::Event::Category":
        """Parameter controlling the behavior of the stream.

        *Values*

         - ``Default``: Default event.
         - ``BlockingSync``: Event meant to be synchronize with CPU (process on CPU blockled until the event occurs).
         - ``DisableTiming``: Event not recording time data.
         - ``EventInterprocess``: Event might be used in an interprocess communication.
        """
        Default,
        BlockingSync,
        DisableTiming,
        EventInterprocess

cdef extern from "merlin/cuda/stream.hpp":

    cpdef enum class StreamSetting "merlin::cuda::Stream::Setting":
        """Parameter controlling the behavior of the stream.

        *Values*

         - ``Default``: Default stream creation flag (synchonized with the null stream).
         - ``NonBlocking``: Works may run concurrently with null stream.
        """
        Default,
        NonBlocking
