cmake_minimum_required(VERSION 3.20)

project(find_cuda_arch LANGUAGES CXX CUDA)

# find CUDAToolkit
include(FindCUDAToolkit)

# add executable
add_executable(cuda_arch_finder "${CMAKE_CURRENT_SOURCE_DIR}/arch_detector.cu")
set_property(TARGET cuda_arch_finder PROPERTY OUTPUT_NAME cuda_arch_finder)
target_compile_options(cuda_arch_finder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
