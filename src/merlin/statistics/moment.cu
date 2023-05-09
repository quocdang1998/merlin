// Copyright 2023 quocdang1998
#include "merlin/statistics/moment.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/utils.hpp"  // merlin::flatten_thread_index, merlin::size_of_block

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Mean
// --------------------------------------------------------------------------------------------------------------------

__cudevice__ double statistics::mean_gpu(const array::Parcel & data, double * buffer) {
    // get thread index and total number of threads
    std::uint64_t thrd_idx = flatten_thread_index();
    std::uint64_t n_th = size_of_block();
    // initialize storing vector
    Vector<double> storing;
    storing.assign(buffer, n_th);
    storing[thrd_idx] = 0.0;
    // initialize index vector
    std::uint64_t * index_buffer_start = reinterpret_cast<std::uint64_t *>(storing.end());
    intvec index;
    index.assign(index_buffer_start + thrd_idx*data.ndim(), data.ndim());
    std::uint64_t size = data.size();
    // add to storing vector
    for (std::uint64_t i_point = thrd_idx; i_point < size; i_point += n_th) {
        contiguous_to_ndim_idx(i_point, data.shape(), index.data());
        storing[thrd_idx] += data[index];
    }
    __syncthreads();
    // reduce the sum
    for (std::uint64_t s = n_th/2; s > 0; s >>= 1) {
        if (thrd_idx < s) {
            storing[thrd_idx] += storing[thrd_idx + s];
        }
        __syncthreads();
    }
    return storing[0];
}

}  // namespace merlin
