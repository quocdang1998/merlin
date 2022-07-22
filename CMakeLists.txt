# =============================================================================
# main project
# =============================================================================

# cmake minimum version
cmake_minimum_required(VERSION 3.21)

# g++ standard
set(CMAKE_CXX_STANDARD 17)

# build project
project(merlin LANGUAGES CXX CUDA) # project name

# source list
list(APPEND MERLIN_SRC
     src/array.cpp)

# options
set(MERLIN_LIBKIND "AUTO"  # -DMERLIN_LIBKIND="STATIC" or "SHARED"
       CACHE STRING "Type of the created library (static or shared/dynamic)")
option(MERLIN_TEST "Build test executables" OFF)  # -DMERLIN_TEST=ON

# =============================================================================
# external packages
# =============================================================================

# git
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
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

# include(CheckLanguage)
# check_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# library and executable
# =============================================================================

# library kind
if(MERLIN_LIBKIND MATCHES "AUTO")
    # auto: dynamic on Linux, static on Windows
    message(STATUS "Auto-decide the kind of library")
    if (UNIX)
        message(STATUS "On Linux, build dynamic library")
        add_library(libmerlin SHARED ${MERLIN_SRC})
    endif (UNIX)
    if (WIN32)
        message(STATUS "On Windows, build static library")
        add_library(libmerlin STATIC ${MERLIN_SRC})
    endif (WIN32)
elseif(MERLIN_LIBKIND MATCHES "STATIC")
    # build static lib
    message(STATUS "Force build static library")
    add_library(libmerlin STATIC ${MERLIN_SRC})
elseif(MERLIN_LIBKIND MATCHES "SHARED" OR MERLIN_LIBKIND MATCHES "DYNAMIC")
    # build dynamic lib
    message(STATUS "Force build dynamic library")
    add_library(libmerlin SHARED ${MERLIN_SRC})
else()
    message(FATAL_ERROR "Library build mode not found")
endif()

# prperty
set_property(TARGET libmerlin PROPERTY OUTPUT_NAME merlin)
set_property(TARGET libmerlin PROPERTY POSITION_INDEPENDENT_CODE 1)
target_include_directories(libmerlin PUBLIC inc/)

# =============================================================================
# tests
# =============================================================================

if (MERLIN_TEST)
    file(GLOB MERLIN_TESTFILES "tests/cpp/*")
    foreach (TESTFILE ${MERLIN_TESTFILES})
        message(STATUS "${TESTFILE}")
        get_filename_component(TESTFILENAME ${TESTFILE} NAME)
        string(REPLACE ".cpp" "" TESTNAME ${TESTFILENAME})
        string(REPLACE ".cu" "" TESTNAME ${TESTFILENAME})
        add_executable(${TESTNAME} ${TESTFILE})
        target_link_libraries(${TESTNAME} libmerlin)
        set_target_properties(${TESTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              ${CMAKE_CURRENT_BINARY_DIR}/tests)
    endforeach(TESTFILE ${MERLIN_TESTFILES})
endif (MERLIN_TEST)
