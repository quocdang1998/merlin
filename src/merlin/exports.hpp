// Copyright 2022 quocdang1998
#ifndef MERLIN_EXPORTS_HPP_
#define MERLIN_EXPORTS_HPP_

#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__

#if defined(__MERLIN_LINUX__)
    #define MERLINSHARED_EXPORTS
#else
    // define MERLINSHARED_EXPORTS to export extern variables, classes and functions to WIndows DLL library
    #ifndef MERLINSHARED_EXPORTS
        #if defined(libmerlinshared_EXPORTS)
            #define MERLINSHARED_EXPORTS __declspec(dllexport)
        #else
            #define MERLINSHARED_EXPORTS __declspec(dllimport)
        #endif  // libmerlinshared_EXPORTS
    #endif  // MERLINSHARED_EXPORTS
#endif  // __MERLIN_LINUX__

#if defined(__MERLIN_BUILT_AS_STATIC__) || defined(__LIBMERLINCUDA__) || defined(__MERLIN_LINUX__)
    #define MERLIN_EXPORTS
    #define MERLIN_HOSTDEV_EXPORTS
    #define MERLIN_NO_EXPORT
#else
    // define MERLIN_EXPORTS to export extern variables, classes and functions to WIndows DLL library
    #ifndef MERLIN_EXPORTS
        #if defined(libmerlin_EXPORTS)
            #define MERLIN_EXPORTS __declspec(dllexport)
            #if !defined(__MERLIN_CUDA__)
                #define MERLIN_HOSTDEV_EXPORTS __declspec(dllexport)
            #else
                #define MERLIN_HOSTDEV_EXPORTS
            #endif  // __MERLIN_CUDA__
        #else
            #define MERLIN_EXPORTS __declspec(dllimport)
            #if !defined(__MERLIN_CUDA__)
                #define MERLIN_HOSTDEV_EXPORTS __declspec(dllimport)
            #else
                #define MERLIN_HOSTDEV_EXPORTS
            #endif  // __MERLIN_CUDA__
        #endif  // libmerlin_EXPORTS
    #endif  // MERLIN_EXPORTS
    // define MERLIN_NO_EXPORT as regular "static" objects
    #ifndef MERLIN_NO_EXPORT
        #define MERLIN_NO_EXPORT
    #endif  // MERLIN_NO_EXPORT
#endif  // __MERLIN_BUILT_AS_STATIC__ || LIBMERLIN_STATIC || __MERLIN_LINUX__

#ifndef MERLIN_DEPRECATED
    #if defined(__MERLIN_WINDOWS__)
        #define MERLIN_DEPRECATED __declspec(deprecated)
    #else
        #define MERLIN_DEPRECATED
    #endif  // __MERLIN_WINDOWS__
#endif

#ifndef MERLIN_DEPRECATED_EXPORT
    #define MERLIN_DEPRECATED_EXPORT MERLIN_EXPORTS MERLIN_DEPRECATED
#endif  // MERLIN_DEPRECATED_EXPORT

#ifndef MERLIN_DEPRECATED_NO_EXPORT
    #define MERLIN_DEPRECATED_NO_EXPORT MERLIN_NO_EXPORT MERLIN_DEPRECATED
#endif  // MERLIN_DEPRECATED_NO_EXPORT

#endif  // MERLIN_EXPORTS_HPP_
