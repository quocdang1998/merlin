#include <cstdio>
#include <vector>

#include "merlin/array.hpp"

int main (void) {
    // initialize an array
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    unsigned int ndim = 2;
    unsigned int dims[2] = {5, 2};
    unsigned int strides[2] = {(unsigned int ) (dims[1] * sizeof(double)),
                               sizeof(double)};
    std::vector<unsigned int> index = {0, 1};
    std::printf("Original array : ");
    for (int i = 0; i < 10; i++) {
        std::printf("%.1f ", A[i]);
    }
    std::printf("\n\n");

    // copy array
    {
        std::printf("Copy an Array object to the first array\n");
        merlin::Array Ar_copy(A, ndim, dims, strides);
        std::printf("Assign value 2.5 to the second element\n");
        Ar_copy[index] = 2.5;
        std::printf("Original:     %.1f\n", A[1]);
        std::printf("Copied array: %.1f\n", Ar_copy[index]);
        std::printf("Expected values for Original(2.0) and Copied(2.5)\n");
    }

    std::printf("\n");

    // assign array
    {
        std::printf("Assign an Array object to the first array\n");
        merlin::Array Ar_assign(A, ndim, dims, strides, false);
        std::printf("Assign value 2.5 to the second element\n");
        Ar_assign[index] = 2.5;
        std::printf("Original:     %.1f\n", A[1]);
        std::printf("Copied array: %.1f\n", Ar_assign[index]);
        std::printf("Expected values for Original(2.5) and Assign(2.5)\n");
    }
}
