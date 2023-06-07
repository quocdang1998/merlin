// Copyright 2023 quocdang1998
#include "merlin/candy/partition.hpp"

#include <cmath>  // std::pow
#include <numeric>  // std::iota
#include <vector>  // std::vector

#include "merlin/utils.hpp"  // merlin::contiguous_to_ndim_idx

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Partition
// --------------------------------------------------------------------------------------------------------------------


// Calculate factorial of an integer
static inline std::uint64_t factorial(std::uint64_t n) {
    std::uint64_t result = 1;
    for (std::uint64_t i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Calculate contiguous index for each partition
static intvec contiguous_index_per_partition(std::uint64_t index, std::uint64_t nparts, std::uint64_t ndim) {
    // calculate number of choice per partition
    intvec num_choice_per_dimension(nparts);
    ndim -= 1;
    for (std::uint64_t i_part = 0; i_part < nparts; i_part++) {
        num_choice_per_dimension[i_part] = std::pow(nparts-i_part, ndim);
    }
    return contiguous_to_ndim_idx(index, num_choice_per_dimension);
}

// Calculate partition index from contiguous index
static intvec index_from_contiguous(std::uint64_t contiguous_index, std::uint64_t i_part,
                                    std::vector<std::vector<std::uint64_t>> & remaining_index) {
    // calculate index vector of partition
    intvec partition_shape(remaining_index.size(), remaining_index[0].size());
    intvec partition_index = contiguous_to_ndim_idx(contiguous_index, partition_shape);
    // get index from remaining index and pop it from intvec array
    intvec result(remaining_index.size()+1);
    for (std::uint64_t i_dim = 0; i_dim < remaining_index.size(); i_dim++) {
        result[i_dim] = remaining_index[i_dim][partition_index[i_dim]];
        remaining_index[i_dim].erase(remaining_index[i_dim].begin() + partition_index[i_dim]);
    }
    result[remaining_index.size()] = i_part;
    return result;
}

// Calculate index
Vector<intvec> candy::partition_index_from_contiguous(std::uint64_t index, std::uint64_t npart, std::uint64_t ndim) {
    // create remaining index storage
    std::vector<std::vector<std::uint64_t>> remaining_index(ndim-1, std::vector<std::uint64_t>(npart));
    for (std::uint64_t i_dim = 0; i_dim < remaining_index.size(); i_dim++) {
        std::iota(remaining_index[i_dim].begin(), remaining_index[i_dim].end(), 0);
    }
    // calculate index for each partition
    intvec index_per_partition = contiguous_index_per_partition(index, npart, ndim);
    Vector<intvec> result(npart);
    for (std::uint64_t i_part = 0; i_part < npart; i_part++) {
        result[i_part] = index_from_contiguous(index_per_partition[i_part], i_part, remaining_index);
    }
    return result;
}

}  // namespace merlin
