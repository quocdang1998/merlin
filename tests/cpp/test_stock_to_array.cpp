#include "merlin/array/stock.hpp"
#include "merlin/array/array.hpp"

#include <chrono>
#include <cstdint>
#include <cinttypes>
#include <thread>

#include "omp.h"
#include <mutex>

int main(void) {
    // initialize array

    // original:
    // [1.0, 3.0, 5.0, 7.0, 9.0 ]
    // [2.0, 4.0, 6.0, 8.0, 10.0]

    // sliced: [:,::2] in Python numpy notation
    // [1.0, 5.0, 9.0 ]
    // [2.0, 6.0, 10.0]

    float A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::uint64_t ndim = 2;
    std::uint64_t dims[2] = {3, 2};
    std::uint64_t strides[2] = {2*(dims[1] * sizeof(float)), sizeof(float)};
    merlin::array::Array Ar(A, ndim, dims, strides);
    Ar.export_to_file("temp.txt");

    std::mutex m;
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        merlin::array::Stock S("temp.txt", 'r');
        merlin::array::Array a = S.to_array();
        m.lock();
        MESSAGE("From thread %d\nNdim: %" PRIu64 ".\nDims: %" PRIu64 " %" PRIu64 ".\n", omp_get_thread_num(), a.ndim(), a.shape()[0], a.shape()[1]);
        for (merlin::array::Array::iterator it = a.begin(); it != a.end(); ++it) {
            std::printf("%f ", a[it.index()]);
        }
        std::printf("\n");
        m.unlock();
    }
}
