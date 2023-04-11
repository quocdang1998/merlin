// Copyright 2022 quocdang1998
#ifndef AP3_MPO_QUERY_MPO_HPP_
#define AP3_MPO_QUERY_MPO_HPP_

#include <string>  // std::string

namespace ap3_mpo {

/** @brief Get geometry names, energy mesh names, isotopes and reactions presenting in the MPO.*/
void query_mpo(const std::string & filename);

}  // namespace ap3_mpo

#endif  // AP3_MPO_QUERY_MPO_HPP_
