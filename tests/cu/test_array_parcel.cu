#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/copy_helpers.hpp"
#include "merlin/cuda/device.hpp"
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
    extern __shared__ char share_ptr[];
    std::uint64_t thread_idx = flatten_thread_index(), block_size = size_of_block();
    int i = 2;
    auto [end_ptr, p_parcel_shr, p_i] = cuda::copy_objects(share_ptr, thread_idx, block_size, *parcel_ptr, i);
    CudaOut("Value from shared memory: %.1f\n", (*p_parcel_shr)[thread_idx]);
    __shared__ double sum;
    if (thread_idx == 0) {
        sum = 0;
    }
    __syncthreads();
    ::atomicAdd_block(&sum, (*p_parcel_shr)[thread_idx]);
    if (thread_idx == 0) {
        CudaOut("Summed value: %.1f\n", sum);
        CudaOut("Integer: %d\n", *p_i);
    }
    __syncthreads();
}

int main(void) {
    // set GPU
    cuda::Device gpu(0);
    gpu.set_as_current();

    // initialize an tensor
    double A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Index dims = {2, 3};
    Index strides = {5*sizeof(double), 2*sizeof(double)};
    array::Array A(A_data, dims, strides, false);
    Message("CPU Array: %s\n", A.str().c_str());

    // copy data to GPU
    cuda::Stream s(cuda::StreamSetting::Default);
    array::Parcel B(A.shape(), s);
    B.transfer_data_to_gpu(A, s);
    Message("GPU Array: %s\n", B.str().c_str());

    // print each element of the tensor
    cuda::Dispatcher mem(s.get_stream_ptr(), B);
    array::Parcel * B_gpu = mem.get<0>();
    print_element<<<1, B.size()>>>(B_gpu);
    print_element_from_shared_memory<<<1, B.size(), B.sharedmem_size() + sizeof(int)>>>(B_gpu);
    A.clone_data_from_gpu(B);
    cuda::Device::synchronize();
}
