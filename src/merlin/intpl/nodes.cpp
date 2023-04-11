// Copyright 2022 quocdang1998
#include "merlin/intpl/nodes.hpp"

#include <cmath>  // std::cos

namespace merlin {

static inline constexpr double pi = 3.14159265358979323846;

// ---------------------------------------------------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------------------------------------------------

// Regular spaced nodes
Vector<double> intpl::linspace_nodes(const double & xmin, const double & xmax, const std::uint64_t & n_point) {
    Vector<double> nodes(n_point);
    for (std::uint64_t i_point = 0; i_point < n_point; i_point++) {
        nodes[i_point] = xmin * (n_point - 1 - i_point) + xmax * i_point;
        nodes[i_point] /= n_point - 1;
    }
    return nodes;
}

// Chebyshev nodes
Vector<double> intpl::chebyshev_nodes(const double & xmin, const double & xmax, const std::uint64_t & n_point) {
    double center = (xmin + xmax) / 2.0;
    double scale = xmin - xmax;
    scale /= 2.0 * std::cos(pi / (2.0 * n_point));
    Vector<double> nodes(n_point);
    for (std::uint64_t i_point = 1; i_point <= n_point; i_point++) {
        nodes[i_point - 1] = center + scale * std::cos(pi * (2.f * i_point - 1.f) / (2.f * n_point));
    }
    return nodes;
}

}  // namespace merlin
