#include <cmath>
#include <cstdint>
#include <cstdio>

#include "merlin/logger.hpp"
#include "merlin/array/array.hpp"  // merlin::Array
#include "merlin/array/operation.hpp"
#include "merlin/utils.hpp"
#include "merlin/array/operation.hpp"

int main(void) {
    // initialize array

    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0 ]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0 ]
    // [2.0, 6.0, 10.0]

    /*merlin::array::Array X({20,16,8});
    X.fill(std::nan(""));
    MESSAGE("Array X: %s\n", X.str().c_str());*/

    MESSAGE("Initialize Array A.\n");
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    merlin::UIntVec dims({3, 2});
    merlin::UIntVec strides({2*(dims[1] * sizeof(double)), sizeof(double)});

    // copy array
    // merlin::Array Ar_copy(A, ndim, dims, strides); // copy using pointer constructor
    merlin::array::Array Ar(A, dims, strides, false);  // copy using copy constructor
    MESSAGE("Original array: %s\n", Ar.str().c_str());
    merlin::array::Array Ar_copy = Ar;

    // print array
    MESSAGE("Expected values : 1.0 2.0 5.0 6.0 9.0 10.0\n");
    MESSAGE("Result          : %s\n", Ar_copy.str().c_str());

    auto [m, v] = Ar.get_mean_variance();
    MESSAGE("Mean-variance   : %f %f\n", m, v);
}
