// Copyright 2022 quocdang1998
#include "merlin/device/gpu_query.hpp"


namespace merlin::device {

#ifndef __MERLIN_CUDA__

int get_device_count(void) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

void print_device_limit(int device) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

bool test_gpu(int device) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA by enabling option MERLIN_CUDA to query for GPU.\n");
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin::device
