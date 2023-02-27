# Copyright 2022 quocdang1998

# =====================================================================================================================
# project
# =====================================================================================================================

# cmake minimum version
cmake_minimum_required(VERSION 3.21)

# options
set(MERLIN_CUDA ON CACHE BOOL "Build library using CUDA")  # -DMERLIN_CUDA=OFF
set(MERLIN_LIBKIND "AUTO"  # -DMERLIN_LIBKIND="STATIC" or "SHARED"
    CACHE INTERNAL "Type of the created library (static or shared)")
set(MERLIN_TEST OFF CACHE BOOL "Build test executables")  # -DMERLIN_TEST=ON

# source list
list(APPEND MERLIN_SRC_CPP
     array/nddata.cpp
     array/array.cpp
     array/parcel.cpp
     array/slice.cpp
     array/stock.cpp
     # src/autodiff.cpp
     cuda/context.cpp
     cuda/device.cpp
     cuda/event.cpp
     cuda/stream.cpp
     filelock.cpp
     iterator.cpp
     interpolant/cartesian_grid.cpp
     interpolant/grid.cpp
     interpolant/lagrange.cpp
     interpolant/newton.cpp
     interpolant/regular_grid.cpp
     interpolant/sparse_grid.cpp
     # interpolant/interpolant.cpp
)
list(APPEND MERLIN_SRC_CU
     array/array.cu
     array/parcel.cu
     cuda/context.cu
     cuda/device.cu
     cuda/event.cu
     cuda/stream.cu
     interpolant/cartesian_grid.cu
     # interpolant/interpolant.cu
)
list(APPEND MERLIN_SRC_RDC
     array/array.rdc
     array/copy.rdc
     array/nddata.rdc
     array/parcel.rdc
     array/slice.rdc
     cuda/device.rdc
     interpolant/cartesian_grid.rdc
     logger.rdc
     utils.rdc
)
list(APPEND MERLIN_SRC_SHR
     env.cpp
)
list(APPEND MERLIN_SRC_SHRCU
     env.cu
)

# convert path to source to absolute path
list(TRANSFORM MERLIN_SRC_CPP PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_CU PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_RDC PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_SHR PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_SHRCU PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)

# project name and language
if(MERLIN_CUDA)
    project(merlin LANGUAGES CXX CUDA)
    set_source_files_properties(${MERLIN_SRC_RDC} PROPERTIES LANGUAGE CUDA)
else()
    project(merlin LANGUAGES CXX)
    set_source_files_properties(${MERLIN_SRC_RDC} PROPERTIES LANGUAGE CXX)
endif(MERLIN_CUDA)

# detect automatically library kind in AUTO mode
if(MERLIN_LIBKIND MATCHES "AUTO")
    # auto: dynamic on Linux, static on Windows
    message(STATUS "Auto-decide the kind of library")
    if(UNIX)
        message(STATUS "On Linux, build dynamic library")
        set(MERLIN_LIBKIND "SHARED")
    endif(UNIX)
    if(MSVC)
        message(STATUS "On Windows, build static library")
        set(MERLIN_LIBKIND "STATIC")
    endif(MSVC)
endif()

# query build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Compile library in debug mode.")
    set(MERLIN_DEBUG ON)
else()
    message(STATUS "Compile library in release mode.")
    set(MERLIN_DEBUG OFF)
endif()

# =====================================================================================================================
# external packages
# =====================================================================================================================

# git
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" OFF)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive \
            failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# CUDA
if(MERLIN_CUDA)
    # set CUDA compile properties
    find_package(CUDA QUIET REQUIRED)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # CUDA Driver library (adapted from CMake/FindCUDAToolkit.cmake#L550-L585)
    find_library(CUDA_DRIVER_LIBRARY NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDA_PATH
                 PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
    if(NOT CUDA_DRIVER_LIBRARY)
        find_library(CUDA_DRIVER_LIBRARY NAMES cuda_driver cuda
                     HINTS ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDA_PATH
                     PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
    endif(NOT CUDA_DRIVER_LIBRARY)
    mark_as_advanced(CUDA_DRIVER_LIBRARY)
    # detect CUDA architechture
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    message(STATUS "Detected CUDA architechtures ${CUDA_ARCH_LIST}")
endif(MERLIN_CUDA)

# OpenMP
find_package(OpenMP)

# =====================================================================================================================
# library
# =====================================================================================================================

# set compilation options
function(MerlinSetTargetProperties target_library)
    set_property(TARGET ${target_library} PROPERTY POSITION_INDEPENDENT_CODE 1)
    set_property(TARGET ${target_library} PROPERTY CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(${target_library} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(MERLIN_CUDA)
        set_property(TARGET ${target_library} PROPERTY CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_compile_definitions(${target_library} PUBLIC __MERLIN_CUDA__)
        set(DIAG_SUPPRESS -Xcudafe "--diag_suppress=field_without_dll_interface")
        set(RELAXED_CONSTEXPR --expt-relaxed-constexpr)
        set(MERLIN_CUDA_OPTION ${DIAG_SUPPRESS} ${RELAXED_CONSTEXPR})
        target_compile_options(${target_library} PUBLIC
                               $<$<COMPILE_LANGUAGE:CUDA>: ${MERLIN_CUDA_OPTION} >)
        target_link_libraries(${target_library} PUBLIC ${CUDA_DRIVER_LIBRARY} ${CUDA_DEVRT_LIBRARY})
    endif(MERLIN_CUDA)
    if(MERLIN_DEBUG)
        target_compile_definitions(${target_library} PUBLIC __MERLIN_DEBUG__)
        if(UNIX)
            target_compile_options(${target_library} PRIVATE -rdynamic)
        endif(UNIX)
        if(MSVC)
            target_link_libraries(${target_library} PRIVATE DbgHelp.lib)
        endif(MSVC)
    endif(MERLIN_DEBUG)
    target_link_libraries(${target_library} PUBLIC OpenMP::OpenMP_CXX)
endfunction(MerlinSetTargetProperties)

# setting library for static variables
if(MERLIN_CUDA)
    add_library(libmerlinshared SHARED ${MERLIN_SRC_SHR} ${MERLIN_SRC_SHRCU})
    set_source_files_properties(${MERLIN_SRC_SHRCU} PROPERTIES LANGUAGE CUDA)
else()
    add_library(libmerlinshared SHARED ${MERLIN_SRC_SHR})
endif(MERLIN_CUDA)
set_source_files_properties(${MERLIN_SRC_SHR} PROPERTIES LANGUAGE CXX)
set_property(TARGET libmerlinshared PROPERTY OUTPUT_NAME merlinshared)
set_property(TARGET libmerlinshared PROPERTY CUDA_SEPARABLE_COMPILATION ON)
MerlinSetTargetProperties(libmerlinshared)

# setting library for device code if compiling a shared library with CUDA
if(MERLIN_CUDA AND MERLIN_LIBKIND MATCHES "SHARED")
    set(BUILD_MERLINCUDA_LIB ON)
    add_library(libmerlincuda STATIC ${MERLIN_SRC_RDC})
    set_property(TARGET libmerlincuda PROPERTY OUTPUT_NAME merlincuda)
    set_property(TARGET libmerlincuda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(libmerlincuda PRIVATE __LIBMERLINCUDA__)
    target_link_libraries(libmerlincuda PUBLIC libmerlinshared)
    MerlinSetTargetProperties(libmerlincuda)
else()
    set(BUILD_MERLINCUDA_LIB OFF)
endif()

# concatenate source lists
list(APPEND MERLIN_SRC ${MERLIN_SRC_CPP})
set_source_files_properties(${MERLIN_SRC_CPP} PROPERTIES LANGUAGE CXX)
if(MERLIN_CUDA)
    list(APPEND MERLIN_SRC ${MERLIN_SRC_CU})
    set_source_files_properties(${MERLIN_SRC_CU} PROPERTIES LANGUAGE CUDA)
    if(NOT BUILD_MERLINCUDA_LIB)
        list(APPEND MERLIN_SRC ${MERLIN_SRC_RDC})
        set_source_files_properties(${MERLIN_SRC_RDC} PROPERTIES LANGUAGE CUDA)
    endif(NOT BUILD_MERLINCUDA_LIB)
else()
    list(APPEND MERLIN_SRC ${MERLIN_SRC_RDC})
    set_source_files_properties(${MERLIN_SRC_RDC} PROPERTIES LANGUAGE CXX)
endif()

# main library
if(MERLIN_LIBKIND MATCHES "STATIC")  # build static lib
    message(STATUS "Build static library")
    add_library(libmerlin STATIC ${MERLIN_SRC})
    target_compile_definitions(libmerlin PUBLIC __MERLIN_BUILT_AS_STATIC__)
elseif(MERLIN_LIBKIND MATCHES "SHARED")  # build dynamic lib
    message(STATUS "Build dynamic library")
    add_library(libmerlin SHARED ${MERLIN_SRC})
else()
    message(FATAL_ERROR "Library build mode not found")
endif()
set_property(TARGET libmerlin PROPERTY OUTPUT_NAME merlin)
target_link_libraries(libmerlin PUBLIC libmerlinshared)
if(MERLIN_CUDA)
    set_property(TARGET libmerlin PROPERTY CUDA_RUNTIME_LIBRARY Static)
    set_property(TARGET libmerlin PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    if(BUILD_MERLINCUDA_LIB)
        target_link_libraries(libmerlin PUBLIC libmerlincuda)
        set_property(TARGET libmerlin PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    endif(BUILD_MERLINCUDA_LIB)
endif(MERLIN_CUDA)
MerlinSetTargetProperties(libmerlin)

# =====================================================================================================================
# external executable
# =====================================================================================================================

# Apollo3 MPO reader
add_subdirectory(ext/ap3_mpo)

# =====================================================================================================================
# tests
# =====================================================================================================================

# build execuatble for each tests/{LANGUAGE}/test_* files
function(CONFIGURE_TESTFILES LANGUAGE)
    message(STATUS "Build unit tests ${LANGUAGE}")
    file(GLOB MERLIN_TESTFILES "tests/${LANGUAGE}/test_*")
    foreach (TESTFILE ${MERLIN_TESTFILES})
        get_filename_component(TESTFILENAME ${TESTFILE} NAME)
        string(REPLACE ".${LANGUAGE}" "" TESTNAME ${TESTFILENAME})
        add_executable(${TESTNAME} ${TESTFILE})
        target_link_libraries(${TESTNAME} libmerlin)
        set_property(TARGET ${TESTNAME} PROPERTY CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
        set_property(TARGET ${TESTNAME} PROPERTY CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        set_property(TARGET ${TESTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        if(BUILD_MERLINCUDA_LIB)
            target_link_libraries(${TESTNAME} libmerlincuda)
        endif(BUILD_MERLINCUDA_LIB)
        set_target_properties(${TESTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              ${CMAKE_CURRENT_BINARY_DIR}/tests)
    endforeach(TESTFILE ${MERLIN_TESTFILES})
endfunction(CONFIGURE_TESTFILES)

# add test executable
if(MERLIN_TEST)
    # C++ test files
    configure_testfiles(cpp)
    # Cu test files
    if(MERLIN_CUDA)
        configure_testfiles(cu)
    endif(MERLIN_CUDA)
endif(MERLIN_TEST)

# =====================================================================================================================
# write config variable to file
# =====================================================================================================================

# initialize Python script embedding CMake config variables
set(MERLIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/setup_cfg/config.py)
file(WRITE ${MERLIN_CONFIG} "")

# CMake_bin_dir
file(APPEND ${MERLIN_CONFIG} "MERLIN_BIN_DIR = \"${CMAKE_BINARY_DIR}\"\n")

# CUDA option
if(MERLIN_CUDA)
    file(APPEND ${MERLIN_CONFIG} "MERLIN_CUDA = True\n")
    file(APPEND ${MERLIN_CONFIG} "NVCC = \"${CUDA_NVCC_EXECUTABLE}\"\n")
    cmake_path(GET CUDA_cudart_static_LIBRARY PARENT_PATH CUDALIB)
    file(APPEND ${MERLIN_CONFIG} "CUDALIB = \"${CUDALIB}\"\n")
    cmake_path(GET CUDA_cudart_static_LIBRARY FILENAME CUDART_STATIC)
    file(APPEND ${MERLIN_CONFIG} "CUDART = \"${CUDART_STATIC}\"\n")
    cmake_path(GET CUDA_cudadevrt_LIBRARY FILENAME CUDADEVRT)
    file(APPEND ${MERLIN_CONFIG} "CUDADEVRT = \"${CUDADEVRT}\"\n")
    cmake_path(GET CUDA_DRIVER_LIBRARY FILENAME CUDADRIVER)
    file(APPEND ${MERLIN_CONFIG} "CUDADRIVER = \"${CUDADRIVER}\"\n")
else()
    file(APPEND ${MERLIN_CONFIG} "MERLIN_CUDA = False\n")
endif(MERLIN_CUDA)

# Debug option
if(MERLIN_DEBUG)
    file(APPEND ${MERLIN_CONFIG} "MERLIN_DEBUG = True\n")
else()
    file(APPEND ${MERLIN_CONFIG} "MERLIN_DEBUG = False\n")
endif(MERLIN_DEBUG)

# Lib kind
file(APPEND ${MERLIN_CONFIG} "MERLIN_LIBKIND = \"${MERLIN_LIBKIND}\"\n")
