// Copyright 2022 quocdang1998
#include "merlin/array/nddata.hpp"

#include <cstring>  // std::memcpy
#include <numeric>  // std::iota

#include "merlin/logger.hpp"
#include "merlin/array/utils.hpp"

int main(void) {
    MESSAGE("Test NdData class.\n");
    float A_ptr[6];
    std::iota<float *, float>(A_ptr, A_ptr + 6, 0.0);  // A = {0, ..., 5}
    merlin::array::NdData A_array(A_ptr, 2, {2, 3}, {3*sizeof(float), sizeof(float)});
    MESSAGE("NdData A = [[%.1f %.1f %.1f], [%.1f %.1f %.1f]]\n", A_array.data()[0], A_array.data()[1],
            A_array.data()[2], A_array.data()[3], A_array.data()[4], A_array.data()[5]);

    float B_ptr[6];
    merlin::array::NdData B_array(B_ptr, 2, {2, 3}, {3*sizeof(float), sizeof(float)});
    MESSAGE("Initialize NdData B.\n");
    MESSAGE("NdData B = [[%.1f %.1f %.1f], [%.1f %.1f %.1f]]\n", B_array.data()[0], B_array.data()[1],
            B_array.data()[2], B_array.data()[3], B_array.data()[4], B_array.data()[5]);

    MESSAGE("Cloning array B from array A with std::memcpy.\n");
    merlin::array_copy(&B_array, &A_array, std::memcpy);
    MESSAGE("NdData B = [[%.1f %.1f %.1f], [%.1f %.1f %.1f]]\n", B_array.data()[0], B_array.data()[1],
            B_array.data()[2], B_array.data()[3], B_array.data()[4], B_array.data()[5]);
}
