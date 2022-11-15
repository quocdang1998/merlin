// Copyright 2022 quocdang1998
#ifndef MERLIN_SETTINGS_HPP_
#define MERLIN_SETTINGS_HPP_

#include <map>  // std::map

namespace merlin::settings {

/** @brief Map from GPU ID to is details.*/
MERLIN_EXPORTS extern std::map<int, Device *> gpu_map;

}  // namespace merlin::settings

#endif  // MERLIN_SETTINGS_HPP_
