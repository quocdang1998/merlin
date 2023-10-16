# Copyright 2022 quocdang1998

# =====================================================================================================================
# project
# =====================================================================================================================

# cmake minimum version
cmake_minimum_required(VERSION 3.25)

# options
set(MERLIN_CUDA ON CACHE BOOL "Build library using CUDA")  # -DMERLIN_CUDA=OFF
set(MERLIN_DETECT_CUDA_ARCH ON CACHE BOOL
    "Automatically detect CUDA architectures of all GPUs \
    (use CMAKE_CUDA_ARCHITECTURES otherwise)")  # -MERLIN_DETECT_CUDA_ARCH=OFF
set(MERLIN_LIBKIND "AUTO"  # -DMERLIN_LIBKIND="STATIC" or "SHARED"
    CACHE STRING "Type of the created library (static or shared)")
set(MERLIN_TEST OFF CACHE BOOL "Build test executables")  # -DMERLIN_TEST=ON

# source list
list(APPEND MERLIN_SRC_CPP
     array/nddata.cpp
     array/array.cpp
     array/parcel.cpp
     array/stock.cpp
     candy/optmz/adam.cpp
     candy/optmz/adagrad.cpp
     candy/optmz/grad_descent.cpp
     candy/launcher.cpp
     candy/loss.cpp
     candy/model.cpp
     candy/optimizer.cpp
     candy/partition.cpp
     cuda/context.cpp
     cuda/device.cpp
     cuda/event.cpp
     cuda/graph.cpp
     cuda/stream.cpp
     filelock.cpp
     iterator.cpp
     intpl/cartesian_grid.cpp
     intpl/grid.cpp
     intpl/interpolant.cpp
     intpl/lagrange.cpp
     intpl/newton.cpp
     intpl/nodes.cpp
     intpl/regular_grid.cpp
     intpl/sparse_grid.cpp
     linalg/qr_solve.cpp
     linalg/matrix.cpp
     slice.cpp
     shuffle.cpp
     splint/cartesian_grid.cpp
     splint/interpolator.cpp
     splint/tools.cpp
     statistics/moment.cpp
)
list(APPEND MERLIN_SRC_CU
     array/array.cu
     array/parcel.cu
     candy/launcher.cu
     candy/model.cu
     cuda/context.cu
     cuda/device.cu
     cuda/event.cu
     cuda/graph.cu
     cuda/stream.cu
     intpl/cartesian_grid.cu
     intpl/lagrange.cu
     intpl/newton.cu
     splint/cartesian_grid.cu
     statistics/moment.cu
)
list(APPEND MERLIN_SRC_GLB
     candy/optimizer.glb
     candy/optmz/adam.glb
     candy/optmz/adagrad.glb
     candy/optmz/grad_descent.glb
     candy/launcher.glb
     candy/loss.glb
     cuda/device.glb
     intpl/lagrange.glb
     intpl/newton.glb
     splint/tools.glb
)
list(APPEND MERLIN_SRC_RDC
     array/operation.rdc
     array/parcel.rdc
     candy/launcher.rdc
     candy/loss.rdc
     candy/model.rdc
     candy/optimizer.rdc
     candy/optmz/adam.rdc
     candy/optmz/adagrad.rdc
     candy/optmz/grad_descent.rdc
     cuda/device.rdc
     intpl/cartesian_grid.rdc
     intpl/lagrange.rdc
     intpl/newton.rdc
     linalg/qr_solve.rdc
     linalg/matrix.rdc
     slice.rdc
     splint/cartesian_grid.rdc
     splint/intpl/lagrange.rdc
     splint/intpl/linear.rdc
     splint/intpl/newton.rdc
     splint/tools.rdc
     utils.rdc
)
list(APPEND MERLIN_SRC_SHR
     cuda_interface.cpp
     env.cpp
     logger.cpp
)
list(APPEND MERLIN_SRC_SHRCU
     cuda_interface.cu
     env.cu
)

# convert path to source to absolute path
list(TRANSFORM MERLIN_SRC_CPP PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_CU PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_GLB PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_RDC PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_SHR PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_SHRCU PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)

# project name and language
if(MERLIN_CUDA)
    project(merlin LANGUAGES CXX CUDA)
    set_source_files_properties(${MERLIN_SRC_RDC} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${MERLIN_SRC_GLB} PROPERTIES LANGUAGE CUDA)
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
    message(STATUS "Compile library in debug mode")
    set(MERLIN_DEBUG ON)
else()
    message(STATUS "Compile library in release mode")
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
    if(MERLIN_DETECT_CUDA_ARCH)
        message(STATUS "Automatically detecting CUDA architechture")
        include(FindCUDA/select_compute_arch)
        CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
        string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
        string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
        string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
        message(STATUS "Detected CUDA architechtures ${CUDA_ARCH_LIST} on this machine")
    else()
        message(STATUS "CUDA architechtures was manually set as ${CMAKE_CUDA_ARCHITECTURES}")
        set(CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# OpenMP
find_package(OpenMP QUIET REQUIRED)
if(MSVC)  # patch the MSVC OpenMP default 2.0 version
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
                 INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:-openmp:experimental>)
endif(MSVC)

# =====================================================================================================================
# library
# =====================================================================================================================

# set compilation options
function(MerlinSetTargetProperties target_library)
    set_property(TARGET ${target_library} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target_library} PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
    target_compile_features(${target_library} PUBLIC cxx_std_17 cuda_std_17)
    target_include_directories(
        ${target_library} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )
    set_property(TARGET ${target_library} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    if(MERLIN_CUDA)
        set_property(TARGET ${target_library} PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
        target_compile_definitions(${target_library} PUBLIC __MERLIN_CUDA__)
        target_compile_options(${target_library} PUBLIC
                               $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_link_libraries(${target_library} PUBLIC ${CUDA_DRIVER_LIBRARY} ${CUDA_DEVRT_LIBRARY})
        set_property(TARGET ${target_library} PROPERTY CUDA_RUNTIME_LIBRARY Static)
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
    target_link_libraries(${target_library} PRIVATE OpenMP::OpenMP_CXX)
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
# set_property(TARGET libmerlinshared PROPERTY EXPORT_PROPERTIES CUDA_SEPARABLE_COMPILATION)

# setting library for device code if compiling a shared library with CUDA
add_library(libmerlinrdc STATIC ${MERLIN_SRC_RDC})
set_property(TARGET libmerlinrdc PROPERTY OUTPUT_NAME merlinrdc)
set_property(TARGET libmerlinrdc PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
set_property(TARGET libmerlinrdc PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(libmerlinrdc PRIVATE __LIBMERLINCUDA__)
target_link_libraries(libmerlinrdc PUBLIC libmerlinshared)
MerlinSetTargetProperties(libmerlinrdc)

# add library of CUDA global function
if(MERLIN_CUDA)
    add_library(libmerlincuda STATIC ${MERLIN_SRC_GLB})
    set_property(TARGET libmerlincuda PROPERTY OUTPUT_NAME merlincuda)
    set_property(TARGET libmerlincuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(libmerlincuda PRIVATE __LIBMERLINCUDA__)
    target_link_libraries(libmerlincuda PUBLIC libmerlinshared libmerlinrdc)
    MerlinSetTargetProperties(libmerlincuda)
endif(MERLIN_CUDA)

# concatenate source lists
list(APPEND MERLIN_SRC ${MERLIN_SRC_CPP})
set_source_files_properties(${MERLIN_SRC_CPP} PROPERTIES LANGUAGE CXX)
if(MERLIN_CUDA)
    list(APPEND MERLIN_SRC ${MERLIN_SRC_CU})
    set_source_files_properties(${MERLIN_SRC_CU} PROPERTIES LANGUAGE CUDA)
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
target_link_libraries(libmerlin PUBLIC libmerlinrdc libmerlinshared)
if(MERLIN_CUDA)
    set_property(TARGET libmerlin PROPERTY CUDA_RUNTIME_LIBRARY Static)
    set_property(TARGET libmerlin PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    if(UNIX)
        target_link_libraries(libmerlin INTERFACE libmerlincuda)
    endif(UNIX)
    if(MSVC)
        target_link_libraries(libmerlin PUBLIC libmerlincuda)
    endif(MSVC)
    if(MERLIN_LIBKIND MATCHES "SHARED")
        set_property(TARGET libmerlin PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    endif()
endif(MERLIN_CUDA)
MerlinSetTargetProperties(libmerlin)

# =====================================================================================================================
# extensions
# =====================================================================================================================

# Apollo3 MPO reader
add_subdirectory(ext/ap3_mpo)

# =====================================================================================================================
# install
# =====================================================================================================================

# install header and template files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
        DESTINATION "include"
        FILES_MATCHING PATTERN "*.[ht]pp")

# install to library to lib
list(APPEND MERLIN_LIBS_LIST libmerlin)
if(MERLIN_CUDA)
    list(APPEND MERLIN_LIBS_LIST libmerlincuda)
endif()
list(APPEND MERLIN_LIBS_LIST libmerlinrdc libmerlinshared)
install(TARGETS ${MERLIN_LIBS_LIST}
        EXPORT limerlin-cmake
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")

# cmake export
install(EXPORT limerlin-cmake
        DESTINATION "lib/cmake"
        FILE libmerlinConfig.cmake)

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
        target_link_libraries(${TESTNAME} PRIVATE libmerlin OpenMP::OpenMP_CXX)
        set_property(TARGET ${TESTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${TESTNAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
        set_target_properties(${TESTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              ${CMAKE_CURRENT_BINARY_DIR}/tests)
        set_property(TARGET ${TESTNAME} PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
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
    string(REPLACE ";" "," CUDA_ARCH_LIST_PYTHON "${CUDA_ARCH_LIST}")
    file(APPEND ${MERLIN_CONFIG} "CUDA_ARCHITECHTURE = [${CUDA_ARCH_LIST_PYTHON}]\n")
    cmake_path(GET CUDA_cudart_static_LIBRARY PARENT_PATH CUDALIB)
    file(APPEND ${MERLIN_CONFIG} "CUDALIB = \"${CUDALIB}\"\n")
    cmake_path(GET CUDA_cudart_static_LIBRARY FILENAME CUDART_STATIC)
    file(APPEND ${MERLIN_CONFIG} "CUDART = \"${CUDART_STATIC}\"\n")
    cmake_path(GET CUDA_cudadevrt_LIBRARY FILENAME CUDADEVRT)
    file(APPEND ${MERLIN_CONFIG} "CUDADEVRT = \"${CUDADEVRT}\"\n")
    cmake_path(GET CUDA_DRIVER_LIBRARY FILENAME CUDADRIVER)
    file(APPEND ${MERLIN_CONFIG} "CUDADRIVER = \"${CUDADRIVER}\"\n")
    file(APPEND ${MERLIN_CONFIG} "CUDA_STANDARD_LIBRARIES = \"${CMAKE_CUDA_STANDARD_LIBRARIES}\"\n")
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
