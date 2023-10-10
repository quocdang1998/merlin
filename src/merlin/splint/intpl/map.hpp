// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTPL_MAP_HPP_
#define MERLIN_SPLINT_INTPL_MAP_HPP_

#include <array>        // std::array
#include <cstdint>      // std::uint64_t
#include <type_traits>  // std::add_pointer

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::splint::intpl {

/** @brief Type of construction methods.*/
using ConstructionMethod =
    std::add_pointer<void(double *, double *, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t)>::type;

/** @brief Array of functor for constructing interpolation coefficients by different methods.*/
MERLIN_EXPORTS extern std::array<ConstructionMethod, 2> construction_func_cpu;

}  // namespace merlin::splint::intpl

#endif  // MERLIN_SPLINT_INTPL_MAP_HPP_
