// Copyright 2022 quocdang1998
#include "merlin/splint/intpl/linear.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Linear interpolation (CPU)
// ---------------------------------------------------------------------------------------------------------------------

// Construct interpolation coefficients by linear interpolation method on CPU
void splint::intpl::construction_linear_cpu(double * coeff, const double * grid_nodes, std::uint64_t shape,
                                            std::uint64_t element_size, std::uint64_t thread_idx,
                                            std::uint64_t n_threads) noexcept {}

}  // namespace merlin
