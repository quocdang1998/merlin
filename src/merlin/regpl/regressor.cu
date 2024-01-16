// Copyright 2024 quocdang1998
#include "merlin/regpl/regressor.hpp"

#include "merlin/cuda/memory.hpp"  // merlin::cuda::Memory

namespace merlin {

// Allocate memory for regressor object on GPU
void regpl::allocate_mem_gpu(const regpl::Polynomial & polynom, regpl::Polynomial *& p_poly, double *& matrix_data,
                             std::uintptr_t stream_ptr) {
    // copy polynomial object to GPU
    cuda::Memory gpu_mem(stream_ptr, polynom);
    p_poly = gpu_mem.get<0>();
    gpu_mem.disown();
    // allocate data on GPU
}

}  // namespace merlin
