// Copyright 2022 quocdang1998
#include "merlin/array.hpp"

#include <cstring>  // std::memcpy
#include <numeric>  // std::iota

#include "merlin/logger.hpp"
#include "merlin/utils.hpp"

int main(void) {
    MESSAGE("Test Array class.\n");
    float A_ptr[6];
    std::iota<float *, float>(A_ptr, A_ptr + 6, 0.0);  // A = {0, ..., 5}
    merlin::Array A_array(A_ptr, 2, {2, 3}, {3*sizeof(float), sizeof(float)});
    MESSAGE("Array A = [[%.1f %.1f %.1f], [%.1f %.1f %.1f]]\n", A_array.data()[0], A_array.data()[1],
            A_array.data()[2], A_array.data()[3], A_array.data()[4], A_array.data()[5]);

    float B_ptr[6];
    merlin::Array B_array(B_ptr, 2, {2, 3}, {3*sizeof(float), sizeof(float)});
    MESSAGE("Initialize Array B.\n");
    MESSAGE("Array B = [[%.1f %.1f %.1f], [%.1f %.1f %.1f]]\n", B_array.data()[0], B_array.data()[1],
            B_array.data()[2], B_array.data()[3], B_array.data()[4], B_array.data()[5]);

    MESSAGE("Cloning array B from array A with std::memcpy.\n");
    merlin::array_copy(&B_array, &A_array, std::memcpy);
    MESSAGE("Array B = [[%.1f %.1f %.1f], [%.1f %.1f %.1f]]\n", B_array.data()[0], B_array.data()[1],
            B_array.data()[2], B_array.data()[3], B_array.data()[4], B_array.data()[5]);
}
