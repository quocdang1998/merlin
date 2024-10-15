# Copyright 2022 quocdang1998

# ======================================================================================================================
# project
# ======================================================================================================================

# cmake minimum version
cmake_minimum_required(VERSION 3.28)

# options
set(MERLIN_CUDA OFF CACHE BOOL "Build library using CUDA") # -DMERLIN_CUDA=OFF
set(MERLIN_DETECT_CUDA_ARCH ON # -DMERLIN_DETECT_CUDA_ARCH=OFF
    CACHE BOOL "Automatically detect CUDA architectures of all GPUs (use CMAKE_CUDA_ARCHITECTURES otherwise)")
set(MERLIN_LIBKIND "AUTO" # -DMERLIN_LIBKIND=STATIC or -DMERLIN_LIBKIND=SHARED
    CACHE STRING "Type of the created library (static or shared)")
set(MERLIN_TEST OFF CACHE BOOL "Build test executables") # -DMERLIN_TEST=ON
set(MERLIN_EXT "" CACHE STRING "Space-separated list of extensions to be built") # -DMERLIN_EXT="readmpo spgrid"

# package version
set(MERLIN_VERSION "1.7.1")

# source list
list(
    APPEND
    MERLIN_SRC_CPP
    array/nddata.cpp
    array/operation.cpp
    array/array.cpp
    array/parcel.cpp
    array/stock.cpp
    candy/rand/gaussian.cpp
    candy/rand/uniform.cpp
    candy/optmz/adadelta.cpp
    candy/optmz/adam.cpp
    candy/optmz/adagrad.cpp
    candy/optmz/grad_descent.cpp
    candy/optmz/rmsprop.cpp
    candy/gradient.cpp
    candy/loss.cpp
    candy/model.cpp
    candy/optimizer.cpp
    candy/randomizer.cpp
    candy/train/cpu_trainer.cpp
    candy/train/gpu_trainer.cpp
    candy/train/trainer_base.cpp
    candy/trainer.cpp
    candy/trial_policy.cpp
    cuda/device.cpp
    cuda/event.cpp
    cuda/graph.cpp
    cuda/stream.cpp
    grid/cartesian_grid.cpp
    grid/regular_grid.cpp
    io/file_lock.cpp
    io/file_pointer.cpp
    io/io_engine.cpp
    linalg/dot.cpp
    linalg/matrix.cpp
    linalg/qrp_decomp.cpp
    linalg/tri_solve.cpp
    logger.cpp
    memory.cpp
    permutation.cpp
    regpl/polynomial.cpp
    regpl/vandermonde.cpp
    regpl/regressor.cpp
    slice.cpp
    splint/interpolator.cpp
    splint/tools.cpp
    utils.cpp)
list(
    APPEND
    MERLIN_SRC_CU
    cuda/device.cu
    cuda/event.cu
    cuda/graph.cu
    cuda/stream.cu
    memory.cu
)
list(APPEND MERLIN_SRC_GLB
    candy/train/gpu_trainer.glb
    cuda/device.glb
    splint/tools.glb
    regpl/regressor.glb
)
list(
    APPEND
    MERLIN_SRC_RDC
    array/parcel.rdc
    candy/optmz/adadelta.rdc
    candy/optmz/adagrad.rdc
    candy/optmz/adam.rdc
    candy/optmz/grad_descent.rdc
    candy/optmz/rmsprop.rdc
    candy/gradient.rdc
    candy/loss.rdc
    candy/model.rdc
    candy/optimizer.rdc
    grid/cartesian_grid.rdc
    grid/regular_grid.rdc
    regpl/polynomial.rdc
    splint/intpl/lagrange.rdc
    splint/intpl/linear.rdc
    splint/intpl/newton.rdc
    splint/tools.rdc
    utils.rdc)
list(APPEND MERLIN_SRC_ENV env.cpp color.cpp version.cpp)

# convert path to source to absolute path
list(TRANSFORM MERLIN_SRC_CPP PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_CU PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_GLB PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_RDC PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)
list(TRANSFORM MERLIN_SRC_ENV PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/merlin/)

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
if(MERLIN_LIBKIND STREQUAL "AUTO")
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
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
if((BUILD_TYPE STREQUAL "DEBUG"))
    message(STATUS "Compile library in debug mode")
    set(MERLIN_DEBUG ON)
else()
    message(STATUS "Compile library in release mode")
    set(MERLIN_DEBUG OFF)
endif()

# ======================================================================================================================
# external packages
# ======================================================================================================================

# git
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive \
            failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# CUDA
if(MERLIN_CUDA)
    include(cmake/FindGPUArch.cmake)
endif(MERLIN_CUDA)

# OpenMP
include(cmake/FindOpenMP.cmake)

# AVX
include(cmake/FindAVX.cmake)

# ======================================================================================================================
# library
# ======================================================================================================================

# set compilation options
function(merlinsettargetproperties target_library)
    set_property(TARGET ${target_library} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${target_library} PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
    target_compile_features(${target_library} PUBLIC cxx_std_20 cuda_std_20)
    target_include_directories(${target_library} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                                                        $<INSTALL_INTERFACE:include>)
    set_property(TARGET ${target_library} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    target_compile_options(${target_library} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${AVX_COMPILE_OPTION}>)
    if(MERLIN_CUDA)
        set_property(TARGET ${target_library} PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
        target_compile_definitions(${target_library} PRIVATE __MERLIN_CUDA__)
        target_compile_options(${target_library} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_property(TARGET ${target_library} PROPERTY CUDA_RUNTIME_LIBRARY Static)
        set_property(TARGET ${target_library} APPEND PROPERTY EXPORT_PROPERTIES CUDA_ARCHITECTURES)
    endif(MERLIN_CUDA)
    if(MERLIN_DEBUG)
        if(UNIX)
            target_compile_options(${target_library} PRIVATE -rdynamic)
        endif(UNIX)
        if(MSVC)
            target_link_libraries(${target_library} PRIVATE DbgHelp.lib)
        endif(MSVC)
    endif(MERLIN_DEBUG)
endfunction(merlinsettargetproperties)

# setting library for static variables
add_library(libmerlinenv SHARED ${MERLIN_SRC_ENV})
set_source_files_properties(${MERLIN_SRC_ENV} PROPERTIES LANGUAGE CXX)
set_property(TARGET libmerlinenv PROPERTY OUTPUT_NAME merlinenv)
target_compile_definitions(libmerlinenv PRIVATE __MERLIN_VERSION__="${MERLIN_VERSION}")
merlinsettargetproperties(libmerlinenv)

# setting library for device code if compiling a shared library with CUDA
add_library(libmerlinrdc STATIC ${MERLIN_SRC_RDC})
set_property(TARGET libmerlinrdc PROPERTY OUTPUT_NAME merlinrdc)
set_property(TARGET libmerlinrdc PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
set_property(TARGET libmerlinrdc PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(libmerlinrdc PRIVATE __LIBMERLINCUDA__)
target_link_libraries(libmerlinrdc PUBLIC libmerlinenv)
merlinsettargetproperties(libmerlinrdc)

# add library of CUDA global function
if(MERLIN_CUDA)
    add_library(libmerlincuda STATIC ${MERLIN_SRC_GLB})
    set_property(TARGET libmerlincuda PROPERTY OUTPUT_NAME merlincuda)
    set_property(TARGET libmerlincuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(libmerlincuda PRIVATE __LIBMERLINCUDA__)
    target_link_libraries(libmerlincuda PUBLIC libmerlinenv libmerlinrdc)
    merlinsettargetproperties(libmerlincuda)
endif(MERLIN_CUDA)

# concatenate source lists
list(APPEND MERLIN_SRC ${MERLIN_SRC_CPP})
set_source_files_properties(${MERLIN_SRC_CPP} PROPERTIES LANGUAGE CXX)
if(MERLIN_CUDA)
    list(APPEND MERLIN_SRC ${MERLIN_SRC_CU})
    set_source_files_properties(${MERLIN_SRC_CU} PROPERTIES LANGUAGE CUDA)
endif()

# main library
string(TOUPPER "${MERLIN_LIBKIND}" MERLIN_LIBKIND)
if(MERLIN_LIBKIND STREQUAL "STATIC") # build static lib
    message(STATUS "Build static library")
    add_library(libmerlin STATIC ${MERLIN_SRC})
    target_compile_definitions(libmerlin PUBLIC __MERLIN_BUILT_AS_STATIC__)
elseif(MERLIN_LIBKIND STREQUAL "SHARED") # build dynamic lib
    message(STATUS "Build dynamic library")
    add_library(libmerlin SHARED ${MERLIN_SRC})
else()
    message(FATAL_ERROR "Library build mode not found")
endif()
set_property(TARGET libmerlin PROPERTY OUTPUT_NAME merlin)
target_link_libraries(libmerlin PUBLIC libmerlinrdc libmerlinenv)
target_link_libraries(libmerlin PRIVATE OpenMP::OpenMP_CXX)
if(MERLIN_CUDA)
    set_property(TARGET libmerlin PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(libmerlin PRIVATE CUDA::cuda_driver)
    if(UNIX)
        target_link_libraries(libmerlin INTERFACE libmerlincuda)
    endif(UNIX)
    if(MSVC)
        target_link_libraries(libmerlin PUBLIC libmerlincuda)
    endif(MSVC)
    if(MERLIN_LIBKIND STREQUAL "SHARED")
        set_property(TARGET libmerlin PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    endif()
endif(MERLIN_CUDA)
merlinsettargetproperties(libmerlin)
set_property(TARGET libmerlin PROPERTY INSTALL_RPATH "$\{ORIGIN\}")

# ======================================================================================================================
# extensions
# ======================================================================================================================

# convert to list
separate_arguments(MERLIN_EXT)

# Sparse grid
list(FIND MERLIN_EXT "spgrid" SPGRID_EXT_INDEX)
if(${SPGRID_EXT_INDEX} GREATER -1)
    message(STATUS "Build spgrid extension for interpolation on hierarchical grids.")
    add_subdirectory(ext/spgrid)
endif()

# ======================================================================================================================
# export and install
# ======================================================================================================================

# create library list to export/install
list(APPEND MERLIN_LIBS_LIST libmerlin libmerlinrdc libmerlinenv)
if(MERLIN_CUDA)
    list(APPEND MERLIN_LIBS_LIST libmerlincuda)
endif()

# export libraries so that other CMake targets can refer to it using merlin::libmerlin, etc
export(TARGETS ${MERLIN_LIBS_LIST} NAMESPACE merlin:: FILE MerlinTargets.cmake)

# install header and template files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ DESTINATION "include" FILES_MATCHING PATTERN "*.[ht]pp")

# install libraries
install(TARGETS ${MERLIN_LIBS_LIST} EXPORT merlin-config LIBRARY DESTINATION "lib" ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")

# install CMake export to destination folder
install(EXPORT merlin-config DESTINATION "lib/cmake" NAMESPACE merlin::)

# extra commands for finding dependencies
install(CODE "message(STATUS \"Merlin installed with success!\")")

# ======================================================================================================================
# tests
# ======================================================================================================================

# build execuatble for each tests/{LANGUAGE}/test_* files
function(configure_testfiles LANGUAGE)
    message(STATUS "Build unit tests ${LANGUAGE}")
    file(GLOB MERLIN_TESTFILES "tests/${LANGUAGE}/test_*")
    foreach(TESTFILE ${MERLIN_TESTFILES})
        get_filename_component(TESTFILENAME ${TESTFILE} NAME)
        string(REPLACE ".${LANGUAGE}" "" TESTNAME ${TESTFILENAME})
        add_executable(${TESTNAME} ${TESTFILE})
        target_link_libraries(${TESTNAME} PRIVATE libmerlin OpenMP::OpenMP_CXX)
        set_property(TARGET ${TESTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${TESTNAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
        set_target_properties(${TESTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
        set_property(TARGET ${TESTNAME} PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    endforeach(TESTFILE ${MERLIN_TESTFILES})
endfunction(configure_testfiles)

# add test executable
if(MERLIN_TEST)
    # C++ test files
    configure_testfiles(cpp)
    # CUDA test files
    if(MERLIN_CUDA)
        configure_testfiles(cu)
    endif(MERLIN_CUDA)
endif(MERLIN_TEST)

# ======================================================================================================================
# write config variable to file
# ======================================================================================================================

# initialize Python script embedding CMake config variables
set(MERLIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/setup_cfg/config.py)
file(WRITE ${MERLIN_CONFIG} "")

# CMake_bin_dir
file(APPEND ${MERLIN_CONFIG} "MERLIN_BIN_DIR = \"${CMAKE_BINARY_DIR}\"\n")

# CUDA option
if(MERLIN_CUDA)
    file(APPEND ${MERLIN_CONFIG} "MERLIN_CUDA = True\n")
    file(APPEND ${MERLIN_CONFIG} "NVCC = \"${CUDAToolkit_NVCC_EXECUTABLE}\"\n")
    string(REPLACE ";" "," CUDA_ARCH_LIST_PYTHON "${CUDA_ARCH_LIST}")
    file(APPEND ${MERLIN_CONFIG} "CUDA_ARCHITECHTURE = [${CUDA_ARCH_LIST_PYTHON}]\n")
    cmake_path(GET CUDA_cudart_LIBRARY PARENT_PATH CUDARTLOC)
    cmake_path(GET CUDA_cuda_driver_LIBRARY PARENT_PATH CUDALOC)
    file(APPEND ${MERLIN_CONFIG} "CUDADIR = [\"${CUDARTLOC}\", \"${CUDALOC}\"]\n")
    cmake_path(GET CUDA_cudart_static_LIBRARY FILENAME CUDART)
    file(APPEND ${MERLIN_CONFIG} "CUDART = \"${CUDART}\"\n")
    cmake_path(GET CUDA_cudadevrt_LIBRARY FILENAME CUDADEVRT)
    file(APPEND ${MERLIN_CONFIG} "CUDADEVRT = \"${CUDADEVRT}\"\n")
    cmake_path(GET CUDA_cuda_driver_LIBRARY FILENAME CUDADRIVER)
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

# Version
file(APPEND ${MERLIN_CONFIG} "MERLIN_VERSION = \"${MERLIN_VERSION}\"\n")
