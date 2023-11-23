# Copyright 2022 quocdang1998

cdef extern from "merlin/cuda/enum_wrapper.hpp":

    cpdef enum class DeviceLimit "merlin::cuda::DeviceLimit":
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

    cpdef enum class EventCategory "merlin::cuda::EventCategory":
        """Parameter controlling the behavior of the stream.

        *Values*

         - ``DefaultEvent``: Default event.
         - ``BlockingSyncEvent``: Event meant to be synchronize with CPU (process on CPU blockled until the event occurs).
         - ``DisableTimingEvent``: Event not recording time data.
         - ``InterprocessEvent``: Event might be used in an interprocess communication.
        """
        DefaultEvent,
        BlockingSyncEvent,
        DisableTimingEvent,
        InterprocessEvent

    cpdef enum class StreamSetting "merlin::cuda::StreamSetting":
        """Parameter controlling the behavior of the stream.

        *Values*

         - ``Default``: Default stream creation flag (synchonized with the null stream).
         - ``NonBlocking``: Works may run concurrently with null stream.
        """
        Default,
        NonBlocking
