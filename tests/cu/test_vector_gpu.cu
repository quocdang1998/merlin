#include <cinttypes>

#include "merlin/cuda/device.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/env.hpp"
#include "merlin/vector.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

using namespace merlin;

__global__ void print_elems(UIntVec * vecptr) {
    UIntVec & vec = *vecptr;
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    CudaOut("Value of element %" PRIu64 " of integer vector on GPU: %" PRIu64 ".\n", thread_idx, vec[thread_idx]);
}

__global__ void print_elems_shr(UIntVec * vecptr) {
    extern __shared__ UIntVec share_ptr[];
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    vecptr->copy_by_block(share_ptr, share_ptr+1, thread_idx, block_size);
    CudaOut("Value from shared memory: %" PRIu64 ".\n", share_ptr[0][thread_idx]);
}

__global__ void initialize_on_gpu(void) {
    UIntVec test({1, 2, 3});
    for (int i = 0; i < 3; i++) {
        CudaOut("Value from object initialized and freed on global device memory: %" PRIu64 ".\n", test[i]);
    }
}

int main(void) {
    // create Environment
    Environment::init_cuda(0);

    // create UIntVec instance
    UIntVec x({1, 2, 3}), y({7, 8, 9});
    Message("Initialize UIntVec with values: %" PRIu64 " %" PRIu64 " %" PRIu64 ".\n", x[0], x[1], x[2]);
    Message("Values expected from all tests: {1, 2, 3}.\n");
    // allocate and copy UIntVec to GPU
    cuda::Memory m(0, x);
    UIntVec * ptr_x_gpu = m.get<0>();
    // print vector on GPU
    print_elems<<<1, x.size(), 0, 0>>>(ptr_x_gpu);
    print_elems_shr<<<1, x.size(), x.sharedmem_size(), 0>>>(ptr_x_gpu);
    initialize_on_gpu<<<1, 1, 0, 0>>>();
    // copy vector back to CPU
    y.copy_from_gpu(reinterpret_cast<std::uint64_t *>(ptr_x_gpu + 1));
    cuda::Device::synchronize();
    Message("After copied x from GPU: %s.\n", y.str().c_str());
}
