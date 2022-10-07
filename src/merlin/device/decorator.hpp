// Copyright 2022 quocdang1998
#ifndef MERLIN_DEVICE_DECORATOR_HPP_
#define MERLIN_DEVICE_DECORATOR_HPP_

// CUDA decorator expansion when not compiling nvcc
#ifdef __NVCC__
    #define __cuhost__ __host__
    #define __cudevice__ __device__
    #define __cuhostdev__ __host__ __device__
#else
    #define __cuhost__
    #define __cudevice__
    #define __cuhostdev__
#endif

// Define __MERLIN_LIBCUDA__ for compiling device static linking library
#ifdef __MERLIN_BUILT_AS_STATIC__
    #define __MERLIN_LIBCUDA__
    #define __MERLIN_DLLCUDA__
#else
    #ifdef LIBMERLIN_STATIC
        #define __MERLIN_LIBCUDA__
    #else
        #define __MERLIN_DLLCUDA__
    #endif  // LIBMERLIN_STATIC
#endif  // __MERLIN_BUILT_AS_STATIC__

#endif  // MERLIN_DEVICE_DECORATOR_HPP_
