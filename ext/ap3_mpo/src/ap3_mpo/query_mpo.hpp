// Copyright 2022 quocdang1998
#ifndef EXT_AP3_MPO_SRC_AP3_MPO_QUERY_MPO_HPP_
#define EXT_AP3_MPO_SRC_AP3_MPO_QUERY_MPO_HPP_

#include <string>  // std::string

namespace ap3_mpo {

/** @brief Get geometry names, energy mesh names, isotopes and reactions presenting in the MPO.*/
void query_mpo(const std::string & filename);

}  // namespace ap3_mpo

#endif  // EXT_AP3_MPO_SRC_AP3_MPO_QUERY_MPO_HPP_
