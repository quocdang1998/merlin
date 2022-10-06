#include "merlin/array/parcel.hpp"

#include <cstdint>

#include "merlin/array/array.hpp"
#include "merlin/logger.hpp"

__global__ void print_element(merlin::Parcel * parcel_ptr) {
    merlin::Parcel & parcel = *parcel_ptr;
    CUDAOUT("Value of element %d of integer vector on GPU: %3.1f.\n", threadIdx.x, parcel[threadIdx.x]);
}

__global__ void print_element_from_shared_memory(merlin::Parcel * parcel_ptr) {
    extern __shared__ merlin::Parcel share_ptr[];
    parcel_ptr->copy_to_shared_mem(share_ptr, share_ptr+1);
    CUDAOUT("Value from shared memory: %.1f\n", share_ptr[0][blockIdx.x*blockDim.x+threadIdx.x]);
}

int main(void) {
    // initialize an tensor
    float A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::uint64_t dims[2] = {2, 3};
    std::uint64_t strides[2] = {5*sizeof(float), 2*sizeof(float)};
    merlin::Array A(A_data, 2, dims, strides, false);

    // copy data to GPU and print each element of the tensor
    MESSAGE("Initialize Parcel object with elements: 1.0 3.0 5.0 6.0 8.0 10.0.\n");
    merlin::Parcel B(A);
    merlin::Parcel * B_gpu;
    cudaMalloc(&B_gpu, B.malloc_size());
    B.copy_to_gpu(B_gpu, B_gpu+1);
    print_element<<<1,B.size()>>>(B_gpu);
    print_element_from_shared_memory<<<1,B.size(),B.malloc_size()>>>(B_gpu);
    cudaFree(B_gpu);
    cudaDeviceReset();
}
