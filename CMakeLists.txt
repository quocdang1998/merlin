# =============================================================================
# main project
# =============================================================================

# cmake minimum version
cmake_minimum_required(VERSION 3.21)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# source list
list(APPEND MERLIN_SRC_CPP
     src/nddata.cpp
     # src/autodiff.cpp
     # src/grid.cpp
     # src/interpolant.cpp
     src/array.cpp
     src/utils.cpp
     src/parcel.cpp
)
list(APPEND MERLIN_SRC_CU
     src/parcel.cu
     src/array.cu
     src/utils.cu
)

# options
option(MERLIN_CUDA "Build library using CUDA" ON)  # -DMERLIN_CUDA=OFF
set(MERLIN_LIBKIND "AUTO"  # -DMERLIN_LIBKIND="STATIC" or "SHARED"
       CACHE STRING "Type of the created library (static or shared/dynamic)")
option(MERLIN_TEST "Build test executables" ON)  # -DMERLIN_TEST=ON

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
    add_compile_definitions(__MERLIN_CUDA__)
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
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif(MERLIN_CUDA)

# OpenMP
find_package(OpenMP)

# =============================================================================
# library and executable
# =============================================================================

# library kind
if(MERLIN_LIBKIND MATCHES "AUTO")
    # auto: dynamic on Linux, static on Windows
    message(STATUS "Auto-decide the kind of library")
    if (UNIX)
        message(STATUS "On Linux, build dynamic library")
        set(MERLIN_LIBKIND "SHARED")
    endif (UNIX)
    if (MSVC)
        message(STATUS "On Windows, build static library")
        set(MERLIN_LIBKIND "STATIC")
    endif (MSVC)
endif()
if(MERLIN_LIBKIND MATCHES "STATIC")
    # build static lib
    message(STATUS "Build static library")
    add_library(libmerlin STATIC ${MERLIN_SRC})
    set(MERLIN_CUDA_STATIC_LIB OFF)
    add_compile_definitions(__MERLIN_FORCE_STATIC__)
elseif(MERLIN_LIBKIND MATCHES "SHARED" OR MERLIN_LIBKIND MATCHES "DYNAMIC")
    # build dynamic lib
    message(STATUS "Build dynamic library")
    add_library(libmerlin SHARED ${MERLIN_SRC})
    add_library(libmerlincuda STATIC ${MERLIN_SRC_CU})
    set(MERLIN_CUDA_STATIC_LIB ON)
else()
    message(FATAL_ERROR "Library build mode not found")
endif()

# set property for device static library
if (MERLIN_CUDA_STATIC_LIB)
    set_property(TARGET libmerlincuda PROPERTY OUTPUT_NAME merlincuda)
    set_property(TARGET libmerlincuda PROPERTY POSITION_INDEPENDENT_CODE 1)
    target_include_directories(libmerlincuda PUBLIC inc/)
    target_compile_definitions(libmerlincuda PRIVATE __LIBMERLINCUDA_STATIC__)
    target_link_libraries(libmerlincuda PUBLIC OpenMP::OpenMP_CXX)
endif(MERLIN_CUDA_STATIC_LIB)

# set property for main library
set_property(TARGET libmerlin PROPERTY OUTPUT_NAME merlin)
set_property(TARGET libmerlin PROPERTY POSITION_INDEPENDENT_CODE 1)
target_include_directories(libmerlin PUBLIC inc/)
target_link_libraries(libmerlin PUBLIC OpenMP::OpenMP_CXX)
if (MERLIN_CUDA_STATIC_LIB)
    target_link_libraries(libmerlin PUBLIC libmerlincuda)
endif(MERLIN_CUDA_STATIC_LIB)


# =============================================================================
# tests
# =============================================================================

function(CONFIGURE_TESTFILES LANGUAGE)
    file(GLOB MERLIN_TESTFILES "tests/${LANGUAGE}/test_*")
    foreach (TESTFILE ${MERLIN_TESTFILES})
        get_filename_component(TESTFILENAME ${TESTFILE} NAME)
        string(REPLACE ".${LANGUAGE}" "" TESTNAME ${TESTFILENAME})
        add_executable(${TESTNAME} ${TESTFILE})
        ##set_property(TARGET ${TESTNAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(${TESTNAME} libmerlin)
        if (MERLIN_CUDA_STATIC_LIB)
            target_link_libraries(${TESTNAME} libmerlincuda)
        endif(MERLIN_CUDA_STATIC_LIB)
        set_target_properties(${TESTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              ${CMAKE_CURRENT_BINARY_DIR}/tests)
    endforeach(TESTFILE ${MERLIN_TESTFILES})
endfunction()

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

# set(MERLIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml)
# file(WRITE ${MERLIN_CONFIG} "")
# file(APPEND ${MERLIN_CONFIG} "MERLIN_CUDA:${MERLIN_CUDA}\n")
# file(APPEND ${MERLIN_CONFIG} "MERLIN_LIBKIND:${MERLIN_LIBKIND}\n")
