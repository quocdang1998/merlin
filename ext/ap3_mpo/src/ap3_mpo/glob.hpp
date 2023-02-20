// Copyright 2022 quocdang1998
#ifndef GLOB_HPP_
#define GLOB_HPP_

#include <string>  // std::string
#include <vector>  // std::vector

namespace merlin::ext::ap3mpo {

/** @brief Get list of files matching a certain pattern.*/
std::vector<std::string> glob(const std::string & pattern);

}  // namespace merlin::ext::ap3mpo

#endif  // GLOB_HPP_
