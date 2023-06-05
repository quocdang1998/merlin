#include <cstdint>
#include <cstdio>
#include <cinttypes>

#include "merlin/cuda/memory.hpp"
#include "merlin/vector.hpp"  // merlin::intvec
#include "merlin/logger.hpp"  // CUDAOUT

__global__ void print_element(merlin::intvec * vecptr) {
    merlin::intvec & vec = *vecptr;
    CUDAOUT("Value of element %d of integer vector on GPU: %" PRIu64 ".\n", threadIdx.x, vec[threadIdx.x]);
}

__global__ void print_element_from_shared_memory(merlin::intvec * vecptr) {
    extern __shared__ merlin::intvec share_ptr[];
    vecptr->copy_to_shared_mem(share_ptr, share_ptr+1);
    CUDAOUT("Value from shared memory: %" PRIu64 ".\n", share_ptr[0][threadIdx.x]);
}

__global__ void initialize_intvec_on_gpu(void) {
    merlin::intvec test({1, 2, 3});
    for (int i = 0; i < 3; i++) {
        CUDAOUT("Value from object initialized and freed on global device memory: %" PRIu64 ".\n", test[i]);
    }
}

int main(void) {
    // create intvec instance
    merlin::intvec x({1, 2, 3}), y({7, 8, 9});
    MESSAGE("Initialize intvec with values: %" PRIu64 " %" PRIu64 " %" PRIu64 ".\n", x[0], x[1], x[2]);
    // allocate and copy intvec to GPU
    merlin::cuda::Memory m(0, x, y);
    merlin::intvec * ptr_x_gpu = m.get<1>();
    // print vector
    print_element<<<1,x.size()>>>(ptr_x_gpu);
    print_element_from_shared_memory<<<1,x.size(),x.malloc_size()>>>(ptr_x_gpu);
    initialize_intvec_on_gpu<<<1,1>>>();
    cudaError_t err_ = cudaGetLastError();
    if (err_ != cudaSuccess) {
        FAILURE(merlin::cuda_runtime_error, "%s.\n", cudaGetErrorName(err_));
    }
    cudaDeviceSynchronize();
}
