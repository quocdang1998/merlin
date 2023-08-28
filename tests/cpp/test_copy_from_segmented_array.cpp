#include <cmath>
#include <cstdint>
#include <cstdio>

#include "merlin/logger.hpp"
#include "merlin/array/array.hpp"  // merlin::Array

int main(void) {
    // initialize array

    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0 ]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0 ]
    // [2.0, 6.0, 10.0]

    merlin::array::Array X({1024,256,8});
    X.fill(std::nan(""));

    MESSAGE("Initialize Array A.\n");
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    merlin::intvec dims({3, 2});
    merlin::intvec strides({2*(dims[1] * sizeof(double)), sizeof(double)});

    // copy array
    MESSAGE("Copy Array A to Ar_copy.\n");
    // merlin::Array Ar_copy(A, ndim, dims, strides); // copy using pointer constructor
    merlin::array::Array Ar(A, dims, strides);  // copy using copy constructor
    merlin::array::Array Ar_copy = Ar;

    // print array
    MESSAGE("Expected values : 1.0 2.0 5.0 6.0 9.0 10.0\n");
    MESSAGE("Result          : ");
    /*
    for (merlin::array::Array::iterator it = Ar_copy.begin(); it != Ar_copy.end(); ++it) {
        std::printf("%.1f ", Ar_copy[it.index()]);
    }
    */
    for (double & value : Ar_copy) {
        std::printf("%.1f ", value);
    }
    std::printf("\n");
}
