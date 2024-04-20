# find CUDAToolkit
include(FindCUDAToolkit)

# detect CUDA architechture
if(MERLIN_DETECT_CUDA_ARCH)
    message(STATUS "Automatically detecting CUDA architechture")
    include(FindCUDA/select_compute_arch)
    cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    message(STATUS "Detected CUDA architechtures ${CUDA_ARCH_LIST} on this machine")
else()
    message(STATUS "CUDA architechtures was manually set as ${CMAKE_CUDA_ARCHITECTURES}")
    set(CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# find CUDA device runtime library
_CUDAToolkit_find_and_add_import_lib(cudadevrt ALT cudadevrt DEPS cudart_static_deps)
