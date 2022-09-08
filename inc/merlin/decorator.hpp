// Copyright 2022 quocdang1998
#ifndef MERLIN_DECORATOR_HPP_
#define MERLIN_DECORATOR_HPP_

#ifdef __NVCC__
    #define __cuhost__ __host__
    #define __cudevice__ __device__
    #define __cuhostdev__ __host__ __device__
#else
    #define __cuhost__
    #define __cudevice__
    #define __cuhostdev__
#endif

#endif  // MERLIN_DECORATOR_HPP_
