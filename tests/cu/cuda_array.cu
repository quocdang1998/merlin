#include <cstdio>

#include "merlin/array.hpp"

// function printing elements of a CUDA array
__global__ void print_gpu_array(float * gpu_data) {
    std::printf("GPU element at %d is %f.\n", (blockIdx.x*blockDim.x+threadIdx.x),
                gpu_data[blockIdx.x*blockDim.x+threadIdx.x]);
}

// function double elements of a CUDA array
__global__ void double_element(float * gpu_data) {
    gpu_data[blockIdx.x*blockDim.x+threadIdx.x] *= 2;
}

int main(void) {
    // initialize an array
    float A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    unsigned int dims[2] = {2, 3};
    unsigned int strides[2] = {5*sizeof(float), 2*sizeof(float)};
    merlin::Array A(A_data, 2, dims, strides, false);

    // copy data to GPU and print each element of the array
    A.sync_to_gpu();
    print_gpu_array<<<1,A.size()>>>(A.gpu_data());
    cudaDeviceSynchronize();

    // expected result [[1,3,5], [6,8,10]]
    std::printf("Expected result: 1.0 3.0 5.0 6.0 8.0 10.0\n");

    // doubling result
    double_element<<<1,A.size()>>>(A.gpu_data());
    A.sync_from_gpu();
    std::printf("After doubling, each element of array is: ");
    for (merlin::Array::iterator it = A.begin(); it != A.end(); it++) {
        std::printf("%.1f ", A[it.index()]);
    }
    std::printf("\n");
}
