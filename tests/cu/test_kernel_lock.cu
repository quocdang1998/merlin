#include <cstdio>
#include "merlin/device/lock.hpp"

#define NUMBLOCKS  32
#define NUMTHREADS 8

// each thread 0 of block increases the value of count by 1
// data race will result in undefined behavior
__global__ void increment_without_lock(int * count) {
    if (threadIdx.x == 0) {
        *count += 1;
    }
    __syncthreads();
}

// each thread 0 of block increases sequentially the value of count by 1
// result is the number of block provided to the kernel
__global__ void increment_with_lock(merlin::KernelLock lock, int *numBlocks) {
    lock.lock();
    // only one thread block can enter this region at a time
    if (threadIdx.x == 0) {
        numBlocks[0] = numBlocks[0] + 1;
    }
    lock.unlock();
}

int main(void) {
    int count = 0;
    int * gpu_count;
    cudaMalloc(&gpu_count, sizeof(int));
    merlin::KernelLock lock;

    // test case without lock (data race)
    cudaMemset(gpu_count, 0, sizeof(int));
    increment_without_lock<<<NUMBLOCKS,NUMTHREADS>>>(gpu_count);
    cudaMemcpy(&count, gpu_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::printf("Counting in the unlocked case: %i\n", count);

    // test case with lock (no data race)
    count = 0;
    cudaMemset(gpu_count, 0, sizeof(int));
    increment_with_lock<<<NUMBLOCKS,NUMTHREADS>>>(lock, gpu_count);
    cudaMemcpy(&count, gpu_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Counting in the locked case: %i\n", count);

    cudaFree(gpu_count);
}
