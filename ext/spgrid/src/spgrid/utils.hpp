// Copyright 2023 quocdang1998
#ifndef SPGRID_UTILS_HPP_
#define SPGRID_UTILS_HPP_

#include <cstdint>  // std::uint64_t
#include <set>      // std::set

#include "merlin/array/array.hpp"       // merlin::array::Array
#include "merlin/grid/declaration.hpp"  // merlin::grid::CartesianGrid
#include "merlin/vector.hpp"            // merlin::UIntVec

namespace mln = merlin;

namespace spgrid {

/** @brief Get shape from level.*/
constexpr std::uint64_t shape_from_level(std::uint64_t level) {
    std::uint64_t shape = (level == 0) ? 2 : (1 << (level-1));
    return shape;
}

/** @brief Get max level from a given sparse grid shape.*/
mln::UIntVec get_max_levels(const std::uint64_t * spgrid_shape, std::uint64_t ndim) noexcept;

/** @brief Get index of element in full grid of an hiearchical grid.*/
std::uint64_t get_hiearchical_index(std::uint64_t subgrid_idx, std::uint64_t level,
                                    std::uint64_t fullgrid_shape) noexcept;

/** @brief Get shape of the sub-grid from the level vector.*/
constexpr void ndlevel_to_shape(const mln::UIntVec & ndlevel, std::uint64_t * subgrid_shape_data) noexcept {
    for (std::uint64_t i = 0; i < ndlevel.size(); i++) {
        subgrid_shape_data[i] = shape_from_level(ndlevel[i]);
    }
}

/** @brief Get grid from index.*/
mln::grid::CartesianGrid get_grid(const mln::grid::CartesianGrid & fullgrid,
                                  const mln::Vector<std::set<std::uint64_t>> & idx);

/** @brief Get array from index.*/
mln::array::Array get_data(const mln::array::Array & fulldata, const mln::Vector<std::set<std::uint64_t>> & idx);

}  // namespace spgrid

#endif  // SPGRID_UTILS_HPP_
