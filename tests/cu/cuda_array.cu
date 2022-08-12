#include <cstdio>

#include "merlin/tensor.hpp"

// function printing elements of a CUDA tensor
__global__ void print_gpu_tensor(float * gpu_data) {
    std::printf("GPU element at %d is %f.\n", (blockIdx.x*blockDim.x+threadIdx.x),
                gpu_data[blockIdx.x*blockDim.x+threadIdx.x]);
}

// function double elements of a CUDA tensor
__global__ void double_element(float * gpu_data) {
    gpu_data[blockIdx.x*blockDim.x+threadIdx.x] *= 2;
}

int main(void) {
    // initialize an tensor
    float A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    unsigned int dims[2] = {2, 3};
    unsigned int strides[2] = {5*sizeof(float), 2*sizeof(float)};
    merlin::Tensor A(A_data, 2, dims, strides, false);

    // copy data to GPU and print each element of the tensor
    A.sync_to_gpu();
    print_gpu_tensor<<<1,A.size()>>>(A.gpu_data().back());
    cudaDeviceSynchronize();

    // expected result [[1,3,5], [6,8,10]]
    std::printf("Expected result: 1.0 3.0 5.0 6.0 8.0 10.0\n");

    // doubling result
    double_element<<<1,A.size()>>>(A.gpu_data().back());
    A.sync_from_gpu(A.gpu_data().back());
    std::printf("After doubling, each element of tensor is: ");
    for (merlin::Tensor::iterator it = A.begin(); it != A.end(); it++) {
        std::printf("%.1f ", A[it.index()]);
    }
    std::printf("\n");

    // free the result
    A.free_data_from_gpu();
}
