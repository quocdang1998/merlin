// Copyright 2022 quocdang1998
#ifndef MERLIN_INTPL_NODES_HPP_
#define MERLIN_INTPL_NODES_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::intpl {

/** @brief Regular spaced nodes.*/
Vector<double> linspace_nodes(const double & xmin, const double & xmax, const std::uint64_t & n_point);

/** @brief Chebyshev nodes.*/
Vector<double> chebyshev_nodes(const double & xmin, const double & xmax, const std::uint64_t & n_point);

}  // namespace merlin::intpl

#endif  // MERLIN_INTPL_NODES_HPP_
