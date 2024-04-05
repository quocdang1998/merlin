#include <cstdint>
#include <cstdio>
#include <cinttypes>

#include "merlin/cuda/memory.hpp"
#include "merlin/vector.hpp"  // merlin::UIntVec
#include "merlin/logger.hpp"  // CUDAOUT
#include "merlin/utils.hpp"

__global__ void print_element(merlin::UIntVec * vecptr) {
    merlin::UIntVec & vec = *vecptr;
    CUDAOUT("Value of element %d of integer vector on GPU: %" PRIu64 ".\n", threadIdx.x, vec[threadIdx.x]);
}

__global__ void print_element_from_shared_memory(merlin::UIntVec * vecptr) {
    extern __shared__ merlin::UIntVec share_ptr[];
    std::uint64_t thread_idx = merlin::flatten_thread_index(), block_size = merlin::size_of_block();
    vecptr->copy_by_block(share_ptr, share_ptr+1, thread_idx, block_size);
    CUDAOUT("Value from shared memory: %" PRIu64 ".\n", share_ptr[0][threadIdx.x]);
}

__global__ void initialize_UIntVec_on_gpu(void) {
    merlin::UIntVec test({1, 2, 3});
    for (int i = 0; i < 3; i++) {
        CUDAOUT("Value from object initialized and freed on global device memory: %" PRIu64 ".\n", test[i]);
    }
}

int main(void) {
    // create UIntVec instance
    merlin::UIntVec x({1, 2, 3}), y({7, 8, 9});
    MESSAGE("Initialize UIntVec with values: %" PRIu64 " %" PRIu64 " %" PRIu64 ".\n", x[0], x[1], x[2]);
    // allocate and copy UIntVec to GPU
    merlin::cuda::Memory m(0, x, y);
    merlin::UIntVec * ptr_y_gpu = m.get<1>();
    // print vector
    print_element<<<1,x.size()>>>(ptr_y_gpu);
    print_element_from_shared_memory<<<1,y.size(),y.sharedmem_size()>>>(ptr_y_gpu);
    initialize_UIntVec_on_gpu<<<1,1>>>();
    cudaDeviceSynchronize();
    // copy vector back to CPU
    x.copy_from_gpu(reinterpret_cast<std::uint64_t *>(ptr_y_gpu+1));
    MESSAGE("After copied x from GPU: %s.\n", x.str().c_str());
}
