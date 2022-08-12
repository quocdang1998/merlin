#include <cstdio>
#include <vector>

#include "merlin/tensor.hpp"

int main (void) {
    // initialize an tensor
    float A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    unsigned int ndim = 2;
    unsigned int dims[2] = {5, 2};
    unsigned int strides[2] = {(unsigned int ) (dims[1] * sizeof(float)),
                               sizeof(float)};
    std::vector<unsigned int> index = {0, 1};
    std::printf("Original tensor : ");
    for (int i = 0; i < 10; i++) {
        std::printf("%.1f ", A[i]);
    }
    std::printf("\n\n");

    // copy tensor
    {
        std::printf("Copy an Tensor object to the first tensor\n");
        merlin::Tensor Ar_copy(A, ndim, dims, strides);
        std::printf("Assign value 2.5 to the second element\n");
        Ar_copy[index] = 2.5;
        std::printf("Original:     %.1f\n", A[1]);
        std::printf("Copied tensor: %.1f\n", Ar_copy[index]);
        std::printf("Expected values for Original(2.0) and Copied(2.5)\n");
    }

    std::printf("\n");

    // assign tensor
    {
        std::printf("Assign an Tensor object to the first tensor\n");
        merlin::Tensor Ar_assign(A, ndim, dims, strides, false);
        std::printf("Assign value 2.5 to the second element\n");
        Ar_assign[index] = 2.5;
        std::printf("Original:     %.1f\n", A[1]);
        std::printf("Copied tensor: %.1f\n", Ar_assign[index]);
        std::printf("Expected values for Original(2.5) and Assign(2.5)\n");
    }
}
