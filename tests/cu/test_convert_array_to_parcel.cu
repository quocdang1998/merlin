#include <cstdint>
#include <cstdio>

#include "merlin/logger.hpp"  // MESSAGE
#include "merlin/array/array.hpp"  // merlin::Array
#include "merlin/array/parcel.hpp"  // merlin::Parcel

// function printing elements of a CUDA tensor
__global__ void print_gpu_tensor(double * gpu_data) {
    CUDAOUT("GPU element at %d is %.1f.\n", (blockIdx.x*blockDim.x+threadIdx.x),
            gpu_data[blockIdx.x*blockDim.x+threadIdx.x]);
}

// function double elements of a CUDA tensor
__global__ void double_element(double * gpu_data) {
    gpu_data[blockIdx.x*blockDim.x+threadIdx.x] *= 2;
}

int main(void) {
    // initialize an tensor
    double A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::uint64_t dims[2] = {2, 3};
    std::uint64_t strides[2] = {5*sizeof(double), 2*sizeof(double)};
    merlin::array::Array A(A_data, 2, dims, strides, false);

    // copy data to GPU and print each element of the tensor
    merlin::array::Parcel B(A.shape());
    B.transfer_data_to_gpu(A);
    print_gpu_tensor<<<1,B.size()>>>(B.data());
    cudaDeviceSynchronize();

    // expected result [[1,3,5], [6,8,10]]
    MESSAGE("Expected result: 1.0 3.0 5.0 6.0 8.0 10.0\n");

    // clone result to another Parcel
    merlin::array::Parcel C(B);
    // doubling result
    double_element<<<1,C.size()>>>(C.data());
    A.clone_data_from_gpu(C);
    MESSAGE("After doubling, each element of tensor is: ");
    for (merlin::array::Array::iterator it = A.begin(); it != A.end(); it++) {
        std::printf("%.1f ", A[it.index()]);
    }
    std::printf("\n");
}
