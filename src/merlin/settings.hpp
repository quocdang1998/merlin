#ifndef MERLIN_SETTINGS_HPP_
#define MERLIN_SETTINGS_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t

namespace merlin {

inline constexpr const std::uint64_t max_dim = 16;

using Index = std::array<std::uint64_t, max_dim>;

using Point = std::array<double, max_dim>;

using DPtrArray = std::array<double *, max_dim>;

}  // namespace merlin

#endif  // MERLIN_SETTINGS_HPP_
