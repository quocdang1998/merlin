#include <cstdio>
#include <vector>

#include "merlin/logger.hpp"
#include "merlin/tensor.hpp"

int main(void) {
    // initialize tensor

    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0 ]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0 ]
    // [2.0, 6.0, 10.0]

    MESSAGE("Initialize Tensor A.\n");
    float A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    unsigned int ndim = 2;
    unsigned int dims[2] = {3, 2};
    unsigned int strides[2] = {(unsigned int) (2*(dims[1] * sizeof(float))),
                               sizeof(float)};

    // copy tensor
    MESSAGE("Copy Tensor A to Ar_copy.\n");
    // merlin::Tensor Ar_copy(A, ndim, dims, strides); // copy using pointer constructor
    merlin::Tensor Ar(A, ndim, dims, strides);  // copy using copy constructor
    merlin::Tensor Ar_copy = std::move(Ar);

    // print tensor
    MESSAGE("Expected values : 1.0 2.0 5.0 6.0 9.0 10.0\n");
    MESSAGE("Result          : ");
    for (merlin::Tensor::iterator it = Ar_copy.begin(); it != Ar_copy.end(); ++it) {
        std::printf("%.1f ", Ar_copy[it.index()]);
    }
    std::printf("\n");
}
