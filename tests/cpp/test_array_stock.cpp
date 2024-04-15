#include "merlin/array/stock.hpp"
#include "merlin/array/array.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

#include <chrono>
#include <cstdint>
#include <cinttypes>
#include <thread>

#include "omp.h"
#include <mutex>

int main(void) {
    using namespace merlin;

    // initialize array
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    UIntVec dims = {3, 2};
    UIntVec strides = {2*(dims[1] * sizeof(double)), sizeof(double)};
    array::Array Ar(A, dims, strides, false);
    Message("CPU array: %s\n", Ar.str().c_str());
    {
        array::Stock Stk("temp.txt", Ar.shape());
        Stk.record_data_to_file(Ar);
        Message("Stock array: %s\n", Stk.str().c_str());
    }

    std::mutex m;
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        array::Stock S("temp.txt");
        array::Array Ar_read(S.shape());
        Ar_read.extract_data_from_file(S);
        m.lock();
        Message("From thread %d: %s\n", omp_get_thread_num(), Ar_read.str().c_str());
        m.unlock();
    }

}
