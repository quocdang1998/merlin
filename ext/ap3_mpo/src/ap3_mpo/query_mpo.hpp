// Copyright 2022 quocdang1998
#ifndef QUERY_MPO_HPP_
#define QUERY_MPO_HPP_

#include <string>  // std::string

namespace merlin::ext::ap3mpo {

/** @brief Get geometry names, energy mesh names, isotopes and reactions presenting in the MPO.*/
void query_mpo(const std::string & filename);

}

#endif  // QUERY_MPO_HPP_
