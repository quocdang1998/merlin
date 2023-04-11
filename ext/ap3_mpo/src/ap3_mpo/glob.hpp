// Copyright 2022 quocdang1998
#ifndef AP3_MPO_GLOB_HPP_
#define AP3_MPO_GLOB_HPP_

#include <string>  // std::string
#include <vector>  // std::vector

namespace ap3_mpo {

/** @brief Get list of files matching a certain pattern.*/
std::vector<std::string> glob(const std::string & pattern);

}  // namespace ap3_mpo

#endif  // AP3_MPO_GLOB_HPP_
