// Copyright 2023 quocdang1998
#include "merlin/stat/moment.hpp"

#include <algorithm>   // std::min
#include <cmath>       // std::isnormal
#include <cstdlib>     // std::lldiv
#include <functional>  // std::bind, std::placeholders

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"      // merlin::array::Array
#include "merlin/array/operation.hpp"  // merlin::array::lcseg_and_brindex
#include "merlin/array/stock.hpp"      // merlin::array::Stock
#include "merlin/logger.hpp"           // FAILURE

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Powered Mean
// ---------------------------------------------------------------------------------------------------------------------

// Calculate powered mean on a CPU array
floatvec stat::powered_mean(std::uint64_t order, const array::Array & data, std::uint64_t n_threads) {
    floatvec cache(order * n_threads);
    intvec num_normal_elem(n_threads);
    #pragma omp parallel num_threads(n_threads)
    {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        floatvec cache_thread;
        cache_thread.assign(cache.data() + order * thread_idx, order);
        for (std::uint64_t i = thread_idx; i < data.size(); i += n_threads) {
            double value = data[i];
            if (!std::isnormal(value)) {
                continue;
            }
            for (std::uint64_t j = 0; j < order; j++) {
                cache_thread[j] += value;
                value *= value;
            }
            num_normal_elem[thread_idx]++;
        }
    }
    std::uint64_t num_normals = 0;
    for (std::uint64_t t = 0; t < n_threads; t++) {
        num_normals += num_normal_elem[t];
    }
    floatvec pwr_mean(order);
    for (std::uint64_t t = 0; t < n_threads; t++) {
        for (std::uint64_t j = 0; j < order; j++) {
            pwr_mean[j] += cache[order * t + j] / num_normals;
        }
    }
    return pwr_mean;
}

#ifndef __MERLIN_CUDA__

// Calculate powered mean on a GPU array
floatvec stat::powered_mean(std::uint64_t order, const array::Parcel & data, std::uint64_t n_threads,
                            const cuda::Stream & stream) {
    FAILURE(cuda_compile_error, "Compile merlin with CUDA option to enable this feature.\n");
    return floatvec();
}

#endif  // __MERLIN_CUDA__

}  // namespace merlin
