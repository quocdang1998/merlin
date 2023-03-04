// Copyright 2022 quocdang1998
#ifndef EXT_AP3_MPO_SRC_AP3_MPO_GLOB_HPP_
#define EXT_AP3_MPO_SRC_AP3_MPO_GLOB_HPP_

#include <string>  // std::string
#include <vector>  // std::vector

namespace ap3_mpo {

/** @brief Get list of files matching a certain pattern.*/
std::vector<std::string> glob(const std::string & pattern);

}  // namespace ap3_mpo

#endif  // EXT_AP3_MPO_SRC_AP3_MPO_GLOB_HPP_
