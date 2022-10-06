// Copyright 2022 quocdang1998
#ifndef MERLIN_SETTINGS_HPP_
#define MERLIN_SETTINGS_HPP_

#include <cstdint>  // std::uint64_t
#include <map>  // std::map

namespace merlin {

namespace settings {

/** @brief Max size (in bytes) of merlin::Parcel data on each GPU.*/
extern std::map<int, std::uint64_t> max_gpu_memsize;

/** @brief Max size of merlin::Array.*/
extern std::uint16_t max_memsize;

}  // namespace settings

}  // namespace merlin

#endif  // MERLIN_SETTINGS_HPP_
