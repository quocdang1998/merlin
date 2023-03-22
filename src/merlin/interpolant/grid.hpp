// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_GRID_HPP_
#define MERLIN_INTERPOLANT_GRID_HPP_

#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list

#include "merlin/array/nddata.hpp"  // merlin::NdData
#include "merlin/cuda_decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::interpolant {
class Grid;  // Base grid
class RegularGrid;  // A set of points in multi-dimension
class CartesianGrid;  // Grid formed by Cartesian productions of vectors of points
class SparseGrid;  // Sparse Grid (basic form)
}

namespace merlin {

/** @brief A base class for all kinds of Grid.*/
class interpolant::Grid {
  public:
    /** @brief Default constructor.*/
    __cuhostdev__ Grid(void) {}
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Grid(void);

  protected:
    /** @brief Array holding coordinates of points in the Grid.*/
    array::NdData * points_ = nullptr;
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_GRID_HPP_
