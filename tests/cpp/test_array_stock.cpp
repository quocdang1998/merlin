#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <mutex>
#include <thread>

#include <omp.h>

#include "merlin/array/array.hpp"
#include "merlin/array/stock.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

int main(void) {
    // initialize array
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    Index dims = {3, 2};
    Index strides = {2*(dims[1] * sizeof(double)), sizeof(double)};
    array::Array Ar(A, dims, strides, false);
    Message("CPU array: {}\n", Ar.str());
    {
        array::Stock Stk("temp.txt", Ar.shape(), 0, true);
        Stk.record_data_to_file(Ar);
        Message("Stock array: {}\n", Stk.str());
    }

    _Pragma("omp parallel for") for (int i = 0; i < 10; i++) {
        array::Stock S("temp.txt", 0, true);
        array::Array Ar_read(S.shape());
        Ar_read.extract_data_from_file(S);
        Message("From thread {}: ", omp_get_thread_num()) << Ar_read.str() << "\n";
    }
}
