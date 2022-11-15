// Copyright 2022 quocdang1998
#ifndef MERLIN_PLATFORM_HPP_
#define MERLIN_PLATFORM_HPP_

#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
    #define __MERLIN_WINDOWS__
#elif defined(__linux__) && !defined(__ANDROID__) && defined(__GNUG__)  && !defined(__clang__) && !defined(__INTEL_COMPILER)
    #define __MERLIN_LINUX__
#else
    #error "Platforms other than MSVC on Windows and GNU on Linux are not supported."
#endif

#endif  // MERLIN_PLATFORM_HPP_
