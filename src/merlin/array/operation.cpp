// Copyright 2022 quocdang1998
#include "merlin/array/operation.hpp"

#include <cmath>  // std::isnormal

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Mean and variance
// ---------------------------------------------------------------------------------------------------------------------

// Calculate mean and second moment of a vector
void array::calc_mean_variance(const double * data, std::uint64_t size, double & mean, double & second_moment,
                               std::uint64_t & normal_count) {
    // initialization
    normal_count = 0;
    mean = 0.0;
    second_moment = 0.0;
    // special case: size  = 0
    if (size == 0) {
        return;
    }
    // first pass: calculate mean and count number of non-zeros elements
    for (std::uint64_t i = 0; i < size; i++) {
        if (std::isnormal(data[i])) {
            mean += data[i];
            normal_count += 1;
        }
    }
    mean /= normal_count;
    // second pass: calculate variance
    for (std::uint64_t i = 0; i < size; i++) {
        if (std::isnormal(data[i])) {
            second_moment += (data[i] - mean) * (data[i] - mean);
        }
    }
}

// Combine mean and variance of 2 subsets
void array::combine_stas(double & mean, double & second_moment, std::uint64_t & normal_count,
                         const double & partial_mean, const double & partial_var, std::uint64_t partial_size) {
    std::uint64_t total_count = normal_count + partial_size;
    double total_mean = ((mean * normal_count) + (partial_mean * partial_size)) / total_count;
    second_moment += partial_var;
    second_moment += (normal_count * partial_size * (mean - partial_mean) * (mean - partial_mean)) / total_count;
    mean = total_mean;
    normal_count = total_count;
}

}  // namespace merlin
