// Copyright 2022 quocdang1998
#ifndef MERLIN_PLATFORM_HPP_
#define MERLIN_PLATFORM_HPP_

#if defined(_WIN32) || defined(_WIN64)  // On Windows, detect MSVC
    #if defined(_MSC_VER)
        #define __MERLIN_WINDOWS__
    #elif defined(__MINGW32__) || defined(__MINGW64__)
        #error MinGW on Windows not supported, use MSVC compiler instead.
    #else
        #error Unknown Windows compiler, use MSVC compiler to suppress this error.
    #endif  // _MSC_VER
#elif defined(__linux__) && !defined(__ANDROID__)  // On Linux, detect GCC
    #if defined(__clang__)
        #error Clang not supported, switch to GNU g++ instead.
    #elif defined(__INTEL_COMPILER)
        #error Intel compiler not supported, switch to GNU g++ instead.
    #elif defined(__GNUG__)
        #define __MERLIN_LINUX__
    #else
        #error Unknown Linux compiler, use GNU g++ to suppress this error.
    #endif  // __linux__ && !__ANDROID__
#else
    #error Compilers other than MSVC on Windows and GCC on Linux are not supported.
#endif

#endif  // MERLIN_PLATFORM_HPP_
