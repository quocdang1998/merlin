// Copyright 2022 quocdang1998
#ifndef AP3_MPO_HPP_
#define AP3_MPO_HPP_

#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list
#include <fstream>  // std::fstream
#include <map>  // std::map
#include <string>  // std::string

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/vector.hpp"  // merlin::floatvec

class Ap3HomogXS {
  public:
    Ap3HomogXS(void) = default;
    Ap3HomogXS(const std::string & filename, const std::string & isotope, const std::string & reaction,
               unsigned int energy_group);
    // merge constructor
    // Ap3HomogXS(std::vector<Ap3HomogXS &> & others);


    // merlin::array::Array read_mpo(void);

    ~Ap3HomogXS(void) = default;

  private:
    /** @brief Isotope.*/
    std::string isotope_;
    /** @brief Reaction name.*/
    std::string reaction_;
    /** @brief Energy group.*/
    unsigned int energy_group_;
    /** @brief Parameters.*/
    std::map<std::string, merlin::floatvec> state_param_;
    /** @brief Data to be serialized.*/
    merlin::array::Array data_;
    /** @brief File stream to the exported dataset.*/
    std::fstream file_;
};

#endif  // AP3_MPO_HPP_
