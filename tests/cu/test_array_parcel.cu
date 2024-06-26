#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/env.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

__global__ void print_element(array::Parcel * parcel_ptr) {
    array::Parcel & parcel = *parcel_ptr;
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    CudaOut("Value of element %" PRIu64 " of integer vector on GPU: %3.1f.\n", thread_idx, parcel[thread_idx]);
}

__global__ void print_element_from_shared_memory(array::Parcel * parcel_ptr) {
    extern __shared__ array::Parcel share_ptr[];
    auto [_, __] = cuda::copy_objects(share_ptr, *parcel_ptr);
    CudaOut("Value from shared memory: %.1f\n", share_ptr[0][blockIdx.x*blockDim.x+threadIdx.x]);
    __shared__ double sum;
    if (blockIdx.x*blockDim.x+threadIdx.x == 0) {
        sum = 0;
    }
    __syncthreads();
    ::atomicAdd_block(&sum, share_ptr[0][blockIdx.x*blockDim.x+threadIdx.x]);
    if (blockIdx.x*blockDim.x+threadIdx.x == 0) {
        CudaOut("Summed value: %.1f\n", sum);
    }
    __syncthreads();
}

int main(void) {
    // create Environment
    Environment::init_cuda(0);

    // initialize an tensor
    double A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    UIntVec dims = {2, 3};
    UIntVec strides = {5*sizeof(double), 2*sizeof(double)};
    array::Array A(A_data, dims, strides, false);
    Message("CPU Array: %s\n", A.str().c_str());

    // copy data to GPU and print each element of the tensor
    cuda::Stream s(cuda::StreamSetting::Default);
    array::Parcel B(A.shape(), s);
    B.transfer_data_to_gpu(A, s);
    cuda::Memory mem(s.get_stream_ptr(), B);
    array::Parcel * B_gpu = mem.get<0>();
    print_element<<<1, B.size()>>>(B_gpu);
    print_element_from_shared_memory<<<1, B.size(), B.sharedmem_size()>>>(B_gpu);
    A.clone_data_from_gpu(B);
    cuda::Device::synchronize();
}
