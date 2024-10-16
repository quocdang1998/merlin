#include <cmath>
#include <cstdint>
#include <cstdio>

#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

using namespace merlin;

int main(void) {
    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0 ]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0 ]
    // [2.0, 6.0, 10.0]

    /*array::Array X({20,16,8});
    X.fill(std::nan(""));
    MESSAGE("Array X: {}\n", X.str());*/

    // initialize array
    Message("Initialize Array A.\n");
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    Index dims({3, 2});
    Index strides({2*(dims[1] * sizeof(double)), sizeof(double)});

    // copy array
    // Array Ar_copy(A, ndim, dims, strides); // copy using pointer constructor
    array::Array Ar(A, dims, strides, false);  // copy using copy constructor
    Message("Original array: {}\n", Ar.str());
    array::Array Ar_copy = Ar;

    // print array
    Message("Expected values : 1.0 2.0 5.0 6.0 9.0 10.0\n");
    Message("Result          : {}\n", Ar_copy.str());

    auto [m, v] = Ar.get_mean_variance();

}
