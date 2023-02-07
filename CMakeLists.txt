# Copyright 2022 quocdang1998

# =============================================================================
# project
# =============================================================================

# cmake minimum version
cmake_minimum_required(VERSION 3.21)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# options
set(MERLIN_CUDA ON CACHE BOOL "Build library using CUDA")  # -DMERLIN_CUDA=OFF
set(MERLIN_DEBUG OFF CACHE BOOL "Print stacktrace")  # -DMERLIN_DEBUG=ON
set(MERLIN_LIBKIND "AUTO"  # -DMERLIN_LIBKIND="STATIC" or "SHARED"
    CACHE INTERNAL "Type of the created library (static or shared/dynamic)")
set(MERLIN_TEST ON CACHE BOOL "Build test executables")  # -DMERLIN_TEST=ON

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
     interpolant/regular_grid.cpp
     interpolant/sparse_grid.cpp
     # interpolant/interpolant.cpp
     settings.cpp
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
     interpolant/sparse_grid.rdc
     logger.rdc
     utils.rdc
)

# convert path to source to absolute path
list(TRANSFORM MERLIN_SRC_CPP PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_CU PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_RDC PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)

# project name
if(MERLIN_CUDA)
    project(merlin LANGUAGES CXX CUDA)
else()
    project(merlin LANGUAGES CXX)
endif(MERLIN_CUDA)

# concatenate source lists
list(APPEND MERLIN_SRC ${MERLIN_SRC_CPP})
if(MERLIN_CUDA)
    list(APPEND MERLIN_SRC ${MERLIN_SRC_CU})
endif()

# query build type
if(MERLIN_DEBUG AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "Option debug is only available with build-type Debug")
else()
    message(STATUS "Compile library in debug mode.")
endif()

# =============================================================================
# external packages
# =============================================================================

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
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${MERLIN_SRC_RDC} PROPERTIES LANGUAGE CUDA)
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
else()
    set_source_files_properties(${MERLIN_SRC_RDC} PROPERTIES LANGUAGE CXX)
endif(MERLIN_CUDA)

# OpenMP
find_package(OpenMP)
set(MERLIN_LINKLIBRARY ${MERLIN_LINKLIBRARY} OpenMP::OpenMP_CXX)

# =============================================================================
# library
# =============================================================================

#detect automaticfally library kind in AUTO mode
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

# library kind
if(MERLIN_LIBKIND MATCHES "STATIC")
    # build static lib
    message(STATUS "Build static library")
    list(APPEND MERLIN_SRC ${MERLIN_SRC_RDC})
    add_library(libmerlin STATIC ${MERLIN_SRC})
    target_compile_definitions(libmerlin PUBLIC __MERLIN_BUILT_AS_STATIC__)
elseif(MERLIN_LIBKIND MATCHES "SHARED")
    # build dynamic lib
    message(STATUS "Build dynamic library")
    add_library(libmerlin SHARED ${MERLIN_SRC})
    add_library(libmerlincuda STATIC ${MERLIN_SRC_RDC})
    set(MERLIN_CUDA_STATIC_LIB ON)
else()
    message(FATAL_ERROR "Library build mode not found")
endif()

# add debug linkage option
if(MERLIN_DEBUG)
    if(UNIX)
        set(MERLIN_LINKOPTION ${MERLIN_LINKOPTION} -rdynamic)
    endif(UNIX)
    if(MSVC)
        set(MERLIN_LINKLIBRARY ${MERLIN_LINKLIBRARY} DbgHelp.lib)
    endif(MSVC)
endif(MERLIN_DEBUG)

# set property for host-device static library
if(MERLIN_CUDA_STATIC_LIB)
    set_property(TARGET libmerlincuda PROPERTY OUTPUT_NAME merlincuda)
    set_property(TARGET libmerlincuda PROPERTY POSITION_INDEPENDENT_CODE 1)
    set_property(TARGET libmerlincuda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON
                 CUDA_SEPARABLE_COMPILATION OFF)
    target_include_directories(libmerlincuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(libmerlincuda PRIVATE LIBMERLIN_STATIC)
    if(MERLIN_CUDA)
        target_compile_definitions(libmerlincuda PUBLIC __MERLIN_CUDA__)
        set(MERLIN_CUDA_OPTION --expt-relaxed-constexpr)
        target_compile_options(libmerlincuda PUBLIC
                               $<$<COMPILE_LANGUAGE:CUDA>: ${MERLIN_CUDA_OPTION} >)
        target_link_libraries(libmerlincuda PUBLIC ${CUDA_DRIVER_LIBRARY})
    endif(MERLIN_CUDA)
    if(MERLIN_DEBUG)
        target_compile_definitions(libmerlincuda PUBLIC __MERLIN_DEBUG__)
    endif(MERLIN_DEBUG)
    target_link_libraries(libmerlincuda PUBLIC ${MERLIN_LINKLIBRARY})
    target_link_options(libmerlincuda PUBLIC ${MERLIN_LINKOPTION})
endif(MERLIN_CUDA_STATIC_LIB)

# set property for main library
set_property(TARGET libmerlin PROPERTY OUTPUT_NAME merlin)
set_property(TARGET libmerlin PROPERTY POSITION_INDEPENDENT_CODE 1)
target_include_directories(libmerlin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(libmerlin PUBLIC ${MERLIN_LINKLIBRARY})
target_link_options(libmerlin PUBLIC ${MERLIN_LINKOPTION})
if(MERLIN_CUDA_STATIC_LIB)
    target_link_libraries(libmerlin PUBLIC libmerlincuda)
endif(MERLIN_CUDA_STATIC_LIB)
if(MERLIN_CUDA)
    target_compile_definitions(libmerlin PUBLIC __MERLIN_CUDA__)
    set(DIAG_SUPPRESS -Xcudafe "--diag_suppress=field_without_dll_interface")
    set(RELAXED_CONSTEXPR --expt-relaxed-constexpr)
    set(MERLIN_CUDA_OPTION ${DIAG_SUPPRESS} ${RELAXED_CONSTEXPR})
    target_compile_options(libmerlin PUBLIC
                           $<$<COMPILE_LANGUAGE:CUDA>: ${MERLIN_CUDA_OPTION} >)
    target_link_libraries(libmerlin PUBLIC ${CUDA_DRIVER_LIBRARY})
    set_property(TARGET libmerlin PROPERTY CUDA_RUNTIME_LIBRARY Static)
    if(MERLIN_CUDA_STATIC_LIB)
        set_property(TARGET libmerlin PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    endif(MERLIN_CUDA_STATIC_LIB)
endif(MERLIN_CUDA)
if(MERLIN_DEBUG)
    target_compile_definitions(libmerlin PUBLIC __MERLIN_DEBUG__)
endif(MERLIN_DEBUG)

# =============================================================================
# external executable
# =============================================================================

# Apollo3 MPO reader
find_package(HDF5 COMPONENTS CXX)
list(APPEND AP3_MPO_SRC_CPP
     ext/apollo3/ap3_mpo.cpp
     ext/apollo3/hdf5_utils.cpp
     ext/apollo3/main.cpp
)
add_executable(ap3_mpo ${AP3_MPO_SRC_CPP})
target_include_directories(ap3_mpo PUBLIC ext/apollo3 ${HDF5_INCLUDE_DIRS})
target_link_libraries(ap3_mpo libmerlin ${HDF5_CXX_LIBRARIES} ${HDF5_C_LIBRARIES})

# =============================================================================
# tests
# =============================================================================

# build execuatble for each tests/{LANGUAGE}/test_* files
function(CONFIGURE_TESTFILES LANGUAGE)
    message(STATUS "Build unit tests ${LANGUAGE}")
    file(GLOB MERLIN_TESTFILES "tests/${LANGUAGE}/test_*")
    foreach (TESTFILE ${MERLIN_TESTFILES})
        get_filename_component(TESTFILENAME ${TESTFILE} NAME)
        string(REPLACE ".${LANGUAGE}" "" TESTNAME ${TESTFILENAME})
        add_executable(${TESTNAME} ${TESTFILE})
        target_link_libraries(${TESTNAME} libmerlin)
        if(MERLIN_CUDA_STATIC_LIB)
            target_link_libraries(${TESTNAME} libmerlincuda)
        endif(MERLIN_CUDA_STATIC_LIB)
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

# =============================================================================
# write config variable to file
# =============================================================================

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
