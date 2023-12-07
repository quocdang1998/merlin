#include "merlin/array/parcel.hpp"

#include <cstdint>

#include "merlin/array/array.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

__global__ void print_element(merlin::array::Parcel * parcel_ptr) {
    merlin::array::Parcel & parcel = *parcel_ptr;
    CUDAOUT("Value of element %d of integer vector on GPU: %3.1f.\n", threadIdx.x, parcel[threadIdx.x]);
}

__global__ void print_element_from_shared_memory(merlin::array::Parcel * parcel_ptr) {
    extern __shared__ merlin::array::Parcel share_ptr[];
    auto [_, __] = merlin::cuda::copy_class_to_shared_mem(share_ptr, *parcel_ptr);
    CUDAOUT("Value from shared memory: %.1f\n", share_ptr[0][blockIdx.x*blockDim.x+threadIdx.x]);
    __shared__ double sum;
    if (blockIdx.x*blockDim.x+threadIdx.x == 0) {
        sum = 0;
    }
    __syncthreads();
    ::atomicAdd(&sum, share_ptr[0][blockIdx.x*blockDim.x+threadIdx.x]);
    if (blockIdx.x*blockDim.x+threadIdx.x == 0) {
        CUDAOUT("Summed value: %.1f\n", sum);
    }
    __syncthreads();
}

int main(void) {
    // initialize an tensor
    double A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    merlin::intvec dims = {2, 3};
    merlin::intvec strides = {5*sizeof(double), 2*sizeof(double)};
    merlin::array::Array A(A_data, dims, strides, false);

    // copy data to GPU and print each element of the tensor
    MESSAGE("Initialize Parcel object with elements: 1.0 3.0 5.0 6.0 8.0 10.0.\n");
    merlin::cuda::Stream s(merlin::cuda::StreamSetting::Default);
    merlin::array::Parcel B(A.shape());
    B.transfer_data_to_gpu(A, s);
    merlin::cuda::Memory mem(0, B);
    merlin::array::Parcel * B_gpu = mem.get<0>();
    print_element<<<1,B.size()>>>(B_gpu);
    print_element_from_shared_memory<<<1,B.size(),B.sharedmem_size()>>>(B_gpu);
    cudaDeviceSynchronize();
}
