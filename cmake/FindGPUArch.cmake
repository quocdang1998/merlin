# ======================================================================================================================
# CUDA
# ======================================================================================================================

# find CUDAToolkit
include(FindCUDAToolkit)

# detect CUDA architechture
if(MERLIN_DETECT_CUDA_ARCH)
    message(STATUS "Automatically detecting CUDA architechture")
    # copy file to build directory
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGPUArch
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles)
    # configure, build and execute arch_detector
    execute_process(
        COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_GENERATOR=${CMAKE_GENERATOR} .
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindGPUArch
        OUTPUT_QUIET
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --target cuda_arch_finder
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindGPUArch
        OUTPUT_QUIET
    )
    execute_process(
        COMMAND ./cuda_arch_finder
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindGPUArch
        OUTPUT_VARIABLE CUDA_ARCH_LIST
    )
    message(STATUS "Detected CUDA architechtures ${CUDA_ARCH_LIST} on this machine")
else()
    message(STATUS "CUDA architechtures was manually set as ${CMAKE_CUDA_ARCHITECTURES}")
    set(CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# find CUDA device runtime library
_CUDAToolkit_find_and_add_import_lib(cudadevrt ALT cudadevrt DEPS cudart_static_deps)
