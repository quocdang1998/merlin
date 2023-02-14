#include "merlin/array/stock.hpp"
#include "merlin/array/array.hpp"
#include "merlin/vector.hpp"

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

    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    merlin::intvec dims = {3, 2};
    merlin::intvec strides = {2*(dims[1] * sizeof(double)), sizeof(double)};
    merlin::array::Array Ar(A, dims, strides, false);
    {
        merlin::array::Stock Stk("temp.txt", Ar.shape());
        Stk.record_data_to_file(Ar);
    }

    std::mutex m;
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        merlin::array::Stock S("temp.txt");
        merlin::array::Array Ar_read(S.shape());
        Ar_read.extract_data_from_file(S);
        m.lock();
        MESSAGE("From thread %d\nNdim: %" PRIu64 ".\nDims: %" PRIu64 " %" PRIu64 ".\n", omp_get_thread_num(),
                Ar_read.ndim(), Ar_read.shape()[0], Ar_read.shape()[1]);
        for (merlin::array::Array::iterator it = Ar_read.begin(); it != Ar_read.end(); ++it) {
            std::printf("%.1f ", Ar_read[it.index()]);
        }
        std::printf("\n");
        m.unlock();
    }
}
