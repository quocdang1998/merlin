#include <cstdint>
#include <cstdio>

#include "merlin/logger.hpp"
#include "merlin/array.hpp"  // merlin::Array

int main(void) {
    // initialize array

    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0 ]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0 ]
    // [2.0, 6.0, 10.0]

    MESSAGE("Initialize Array A.\n");
    float A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::uint64_t ndim = 2;
    std::uint64_t dims[2] = {3, 2};
    std::uint64_t strides[2] = {2*(dims[1] * sizeof(float)), sizeof(float)};

    // copy array
    MESSAGE("Copy Array A to Ar_copy.\n");
    // merlin::Array Ar_copy(A, ndim, dims, strides); // copy using pointer constructor
    merlin::Array Ar(A, ndim, dims, strides);  // copy using copy constructor
    merlin::Array Ar_copy = std::move(Ar);

    // print array
    MESSAGE("Expected values : 1.0 2.0 5.0 6.0 9.0 10.0\n");
    MESSAGE("Result          : ");
    for (merlin::Array::iterator it = Ar_copy.begin(); it != Ar_copy.end(); ++it) {
        std::printf("%.1f ", Ar_copy[it.index()]);
    }
    std::printf("\n");
}
