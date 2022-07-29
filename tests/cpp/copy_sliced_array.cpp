#include <cstdio>
#include <vector>

#include "merlin/array.hpp"

int main(void) {
    // initialize array

    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0]
    // [2.0, 6.0, 10.0]

    float A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    unsigned int ndim = 2;
    unsigned int dims[2] = {3, 2};
    unsigned int strides[2] = {(unsigned int) (2*(dims[1] * sizeof(float))),
                               sizeof(float)};

    // copy array
    merlin::Array Ar_copy(A, ndim, dims, strides);

    // print array
    std::printf("Expected values : 1.0 2.0 5.0 6.0 9.0 10.0\n");
    std::printf("Result          : ");
    for (merlin::Array::iterator it = Ar_copy.begin(); it != Ar_copy.end(); ++it) {
        std::printf("%.1f ", Ar_copy[it.index()]);
    }
    std::printf("\n");
}
