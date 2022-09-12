#include "merlin/stock.hpp"
#include "merlin/array.hpp"

int main(void) {
    // initialize array

    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0 ]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0 ]
    // [2.0, 6.0, 10.0]

    float A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    unsigned long int ndim = 2;
    unsigned long int dims[2] = {3, 2};
    unsigned long int strides[2] = {(unsigned int) (2*(dims[1] * sizeof(float))),
                                     sizeof(float)};
    merlin::Array Ar(A, ndim, dims, strides);
    Ar.export_to_file("temp.txt");

    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        merlin::Stock S("temp.txt");
        MESSAGE("Ndim: %lu.\nDims: %lu %lu.\n", S.ndim(), S.shape()[0], S.shape()[1]);
    }
}