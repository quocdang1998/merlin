// Copyright 2023 quocdang1998
#ifndef SPGRID_UTILS_HPP_
#define SPGRID_UTILS_HPP_

#include "merlin/array/nddata.hpp"    // merlin::array::NdData
#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/vector.hpp"          // merlin::intvec

#include "spgrid/declaration.hpp"  // spgrid::SparseGrid

namespace spgrid {

/** @brief Get max level from a given sparse grid shape.*/
merlin::intvec get_max_levels(const merlin::intvec & spgrid_shape) noexcept;

/** @brief Get shape of the sub-grid from the level vector.*/
__cuhostdev__ merlin::intvec ndlevel_to_shape(const merlin::intvec & ndlevel,
                                              std::uint64_t * subgrid_shape_data = nullptr) noexcept;

/** @brief Copy elements from a full Cartesian data into a vector.*/
merlin::floatvec copy_sparsegrid_data_from_cartesian(const merlin::array::NdData full_data, const SparseGrid & grid);

}  // namespace spgrid

#endif  // SPGRID_UTILS_HPP_
