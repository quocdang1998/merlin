// Copyright 2022 quocdang1998
#ifndef HDF5_UTILS_HPP_
#define HDF5_UTILS_HPP_

#include <algorithm>  // std::transform, std::remove_if
#include <cctype>  // std::tolower
#include <cstdint>  // std::uint64_t
#include <string>  // std::string
#include <utility>  // std::pair
#include <vector>  // std::vector

#include "H5Cpp.h"  // H5::Group

#include "merlin/vector.hpp"  // merlin::intvec

inline std::string lowercase(const std::string & s) {
    std::string result(s);
    std::transform(s.begin(), s.end(), result.begin(), [] (char c) {return std::tolower(c);});
    return result;
}

std::uint64_t check_string_in_array(std::string element, std::vector<std::string> array);

std::vector<std::string> ls_groups(H5::Group * group, const char * substring = "");

template <typename T>
std::pair<std::vector<T>, merlin::intvec> get_dset(H5::Group * group, char const * dset_address);

#include "hdf5_utils.tpp"

#endif  // HDF5_UTILS_HPP_
