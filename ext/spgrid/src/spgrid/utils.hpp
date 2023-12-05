// Copyright 2023 quocdang1998
#ifndef SPGRID_UTILS_HPP_
#define SPGRID_UTILS_HPP_

#include "merlin/array/nddata.hpp"      // merlin::array::NdData
#include "merlin/cuda_interface.hpp"    // __cuhostdev__
#include "merlin/grid/declaration.hpp"  // merlin::grid::CartesianGrid
#include "merlin/vector.hpp"            // merlin::intvec

#include "spgrid/declaration.hpp"  // spgrid::SparseGrid

namespace spgrid {

// Get shape from level
inline __cuhostdev__ std::uint64_t shape_from_level(std::uint64_t level) {
    std::uint64_t shape = (level == 0) ? 2 : (1 << (level-1));
    return shape;
}

/** @brief Get max level from a given sparse grid shape.*/
merlin::intvec get_max_levels(const merlin::intvec & spgrid_shape) noexcept;

/** @brief Get index of element in full grid of an hiearchical grid.*/
__cuhostdev__ std::uint64_t get_hiearchical_index(std::uint64_t subgrid_idx, std::uint64_t level,
                                                  std::uint64_t fullgrid_shape) noexcept;

/** @brief Get number of points inside a level.*/
__cuhostdev__ std::uint64_t get_npoint_in_level(const merlin::intvec & ndlevel) noexcept;

/** @brief Get shape of the sub-grid from the level vector.*/
__cuhostdev__ merlin::intvec ndlevel_to_shape(const merlin::intvec & ndlevel,
                                              std::uint64_t * subgrid_shape_data = nullptr) noexcept;

/** @brief Get index in full grid from sub-grid index.
 *  @param subgrid_idx C-contiguous index of the point in the sub-grid.
 *  @param ndlevel Level vector of the sub-grid.
 *  @param fullgrid_shape Shape of the full grid.
 *  @param index_data Pointer to memory storage of the result.
 */
__cuhostdev__ merlin::intvec fullgrid_idx_from_subgrid(std::uint64_t subgrid_idx, const merlin::intvec & ndlevel,
                                                       const merlin::intvec & fullgrid_shape,
                                                       std::uint64_t * index_data = nullptr) noexcept;

/** @brief Copy elements from a full Cartesian data into a vector.*/
merlin::floatvec copy_sparsegrid_data_from_cartesian(const merlin::array::NdData & full_data, const SparseGrid & grid);

/** @brief Merge 2 Cartesian grid.*/
void merge_grid(merlin::grid::CartesianGrid & total_grid, const merlin::grid::CartesianGrid & added_grid) noexcept;

}  // namespace spgrid

#endif  // SPGRID_UTILS_HPP_
