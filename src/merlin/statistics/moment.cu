// Copyright 2023 quocdang1998
#include "merlin/statistics/moment.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/utils.hpp"         // merlin::flatten_thread_index, merlin::size_of_block

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Mean
// ---------------------------------------------------------------------------------------------------------------------

__cudevice__ void statistics::mean_gpu(const array::Parcel & data, double * buffer) {
    // get thread index and total number of threads
    std::uint64_t thread_idx = flatten_thread_index();
    std::uint64_t n_threads = size_of_block();
    // initialize storing vector
    Vector<double> storing;
    storing.assign(buffer, n_threads);
    storing[thread_idx] = 0.0;
    // initialize index vector
    std::uint64_t * index_buffer_start = reinterpret_cast<std::uint64_t *>(storing.end());
    intvec index;
    index.assign(index_buffer_start + thread_idx * data.ndim(), data.ndim());
    // add to storing vector
    for (std::uint64_t i_point = thread_idx; i_point < data.size(); i_point += n_threads) {
        contiguous_to_ndim_idx(i_point, data.shape(), index.data());
        storing[thread_idx] += data[index];
    }
    __syncthreads();
    // reduce the sum
    if (thread_idx < 8) {
        for (std::uint64_t i = thread_idx + 8; i < n_threads; i += 8) {
            storing[thread_idx] += storing[i];
        }
    }
    __syncthreads();
    if (thread_idx == 0) {
        storing[0] += storing[1] + storing[2] + storing[3] + storing[4] + storing[5] + storing[6] + storing[7];
    }
    __syncthreads();
}

}  // namespace merlin
