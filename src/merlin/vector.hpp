// Copyright 2022 quocdang1998
#ifndef MERLIN_VECTOR_HPP_
#define MERLIN_VECTOR_HPP_

#include "merlin/config.hpp"                 // merlin::max_dim
#include "merlin/vector/dynamic_vector.hpp"  // merlin::vector::DynamicVector
#include "merlin/vector/static_vector.hpp"   // merlin::vector::StaticVector
#include "merlin/vector/view.hpp"            // merlin::vector::View

namespace merlin {

// View typedefs
// -------------

/** @brief Range view of unsigned integer values.*/
using UIntView = vector::View<std::uint64_t>;

/** @brief Range view of signed integer values.*/
using IntView = vector::View<std::int64_t>;

/** @brief Range view of floating-points.*/
using DoubleView = vector::View<double>;

// Static vector typedefs
// ----------------------

/** @brief Static vector of unsigned integer values.*/
using Index = vector::StaticVector<std::uint64_t, max_dim>;

/** @brief Static vector of floating-points.*/
using Point = vector::StaticVector<double, max_dim>;

/** @brief Static vector of floating-point arrays.*/
using DPtrArray = vector::StaticVector<double *, max_dim>;

// Dynamic vector typedefs
// -----------------------

/** @brief Dynamic vector of unsigned integer values.*/
using UIntVec = vector::DynamicVector<std::uint64_t>;

/** @brief Dynamic vector of signed integer values.*/
using IntVec = vector::DynamicVector<std::int64_t>;

/** @brief Dynamic vector of floating-points.*/
using DoubleVec = vector::DynamicVector<double>;

// Misc
// ----

/** @brief Static vector of range views of floating-points.*/
using DViewArray = vector::StaticVector<vector::View<double>, max_dim>;

/** @brief Static vector of dynamic vectors of floating-points.*/
using DVecArray = vector::StaticVector<vector::DynamicVector<double>, max_dim>;

}  // namespace merlin

#endif  // MERLIN_VECTOR_HPP_
