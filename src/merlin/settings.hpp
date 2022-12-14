// Copyright 2022 quocdang1998
#ifndef MERLIN_SETTINGS_HPP_
#define MERLIN_SETTINGS_HPP_

#include <cstdint>  // std::uint64_t
#include <map>  // std::map

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::settings {

/** @brief Map from GPU ID to is details.*/
// MERLIN_EXPORTS extern std::map<int, Device *> gpu_map;

/** @brief Memory limit of a process for allocating ``merlin::array::Array``.
 *  @details Default value: 20GB.
*/
MERLIN_EXPORTS extern std::uint64_t cpu_mem_limit;

}  // namespace merlin::settings

#endif  // MERLIN_SETTINGS_HPP_
