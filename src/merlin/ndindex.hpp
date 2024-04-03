#ifndef MERLIN_NDINDEX_HPP_
#define MERLIN_NDINDEX_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t

#include "merlin/cuda_interface.hpp"  // __cuhostdev__

namespace merlin {

inline constexpr const std::uint64_t max_dim = 16;

using Index = std::array<std::uint64_t, max_dim>;

}  // namespace merlin

#endif  // MERLIN_NDINDEX_HPP_
