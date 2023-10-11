// Copyright 2022 quocdang1998
#include "merlin/splint/intpl/map.hpp"

#include "merlin/splint/intpl/linear.hpp"
#include "merlin/splint/intpl/lagrange.hpp"
#include "merlin/splint/intpl/newton.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Map to coefficient construction method (CPU)
// ---------------------------------------------------------------------------------------------------------------------

// Array of functor for constructing interpolation coefficients by different methods
std::array<splint::intpl::ConstructionMethod, 3> splint::intpl::construction_func_cpu {
    splint::intpl::construction_linear_cpu,
    splint::intpl::construction_lagrange_cpu,
    splint::intpl::construction_newton_cpu
};

}  // namespace merlin
