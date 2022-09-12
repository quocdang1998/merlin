// Copyright 2022 quocdang1998
#ifndef MERLIN_EXPORTS_HPP_
#define MERLIN_EXPORTS_HPP_

#if defined(__MERLIN_BUILT_AS_STATIC__) || defined(LIBMERLIN_STATIC) || defined(__GNUG__)
    #define MERLIN_EXPORTS
    #define MERLIN_NO_EXPORT
    #define MERLIN_TEMPLATE_EXPORTS
#else
    // define MERLIN_EXPORTS to export extern variables, classes and functions to WIndows DLL library
    #ifndef MERLIN_EXPORTS
        #if defined(libmerlin_EXPORTS)
            #define MERLIN_EXPORTS __declspec(dllexport)
        #else
            #define MERLIN_EXPORTS __declspec(dllimport)
        #endif  // libmerlin_EXPORTS
    #endif  // MERLIN_EXPORTS
    // define MERLIN_NO_EXPORT as regular "static" objects
    #ifndef MERLIN_NO_EXPORT
        #define MERLIN_NO_EXPORT
    #endif  // MERLIN_NO_EXPORT
#endif  // __MERLIN_BUILT_AS_STATIC__ || __GNUG__

#ifndef MERLIN_DEPRECATED
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define MERLIN_DEPRECATED __declspec(deprecated)
    #else
    #define MERLIN_DEPRECATED
    #endif  // WIN32 || _WIN32 || WIN64 || _WIN64
#endif

#ifndef MERLIN_DEPRECATED_EXPORT
    #define MERLIN_DEPRECATED_EXPORT MERLIN_EXPORTS MERLIN_DEPRECATED
#endif  // MERLIN_DEPRECATED_EXPORT

#ifndef MERLIN_DEPRECATED_NO_EXPORT
    #define MERLIN_DEPRECATED_NO_EXPORT MERLIN_NO_EXPORT MERLIN_DEPRECATED
#endif  // MERLIN_DEPRECATED_NO_EXPORT

#endif  // MERLIN_EXPORTS_HPP_
