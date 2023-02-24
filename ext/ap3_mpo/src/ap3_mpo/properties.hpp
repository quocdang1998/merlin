// Copyright 2022 quocdang1998
#ifndef EXT_AP3_MPO_SRC_AP3_MPO_PROPERTIES_HPP_
#define EXT_AP3_MPO_SRC_AP3_MPO_PROPERTIES_HPP_

#include <cstdint>  // std::uint64_t
#include <map>  // std::map
#include <string>  // std::string
#include <vector>  // std::vector

#include "H5Cpp.h"  // H5::File

#include "merlin/array/stock.hpp"  // merlin::array::Stock

#include "ap3_mpo/declaration.hpp"  // merlin::ext::ap3mpo::Ap3Geometry, merlin::ext::ap3mpo::Ap3EnergyMesh
                                    // merlin::ext::ap3mpo::Ap3StateParam, merlin::ext::ap3mpo::Ap3Isotope

namespace merlin {

struct ext::ap3mpo::Ap3Geometry {
    Ap3Geometry(void) = default;
    Ap3Geometry(const std::string & name, H5::H5File & mpo_file);

    Ap3Geometry(const Ap3Geometry & src) = default;
    ext::ap3mpo::Ap3Geometry & operator=(const ext::ap3mpo::Ap3Geometry & src) = default;
    Ap3Geometry(Ap3Geometry && src) = default;
    ext::ap3mpo::Ap3Geometry & operator=(ext::ap3mpo::Ap3Geometry && src) = default;

    std::string id;
    std::uint64_t index;
    std::string name;
    std::vector<std::string> zone_names;
};

struct ext::ap3mpo::Ap3EnergyMesh {
    Ap3EnergyMesh(void) = default;
    Ap3EnergyMesh(const std::string & name, H5::H5File & mpo_file);

    Ap3EnergyMesh(const Ap3EnergyMesh & src) = default;
    ext::ap3mpo::Ap3EnergyMesh & operator=(const ext::ap3mpo::Ap3EnergyMesh & src) = default;
    Ap3EnergyMesh(Ap3EnergyMesh && src) = default;
    ext::ap3mpo::Ap3EnergyMesh & operator=(ext::ap3mpo::Ap3EnergyMesh && src) = default;

    std::string id;
    std::uint64_t index;
    std::string name;
    std::vector<float> energies;
};

struct ext::ap3mpo::Ap3StateParam {
    Ap3StateParam(void) = default;
    Ap3StateParam(H5::H5File & mpo_file);

    Ap3StateParam(const Ap3StateParam & src) = default;
    ext::ap3mpo::Ap3StateParam & operator=(const ext::ap3mpo::Ap3StateParam & src) = default;
    Ap3StateParam(Ap3StateParam && src) = default;
    ext::ap3mpo::Ap3StateParam & operator=(ext::ap3mpo::Ap3StateParam && src) = default;

    ext::ap3mpo::Ap3StateParam & operator+=(ext::ap3mpo::Ap3StateParam & other);

    std::vector<std::string> param_names;
    std::map<std::string, std::vector<double>> param_values;
    std::vector<std::uint64_t> excluded_index;
};

struct ext::ap3mpo::Ap3Isotope {
    Ap3Isotope(void) = default;
    Ap3Isotope(const std::string & name, H5::H5File & mpo_file);

    Ap3Isotope(const Ap3Isotope & src) = default;
    ext::ap3mpo::Ap3Isotope & operator=(const ext::ap3mpo::Ap3Isotope & src) = default;
    Ap3Isotope(Ap3Isotope && src) = default;
    ext::ap3mpo::Ap3Isotope & operator=(ext::ap3mpo::Ap3Isotope && src) = default;

    std::string name;
    std::uint64_t index;
};

struct ext::ap3mpo::Ap3Reaction {
    Ap3Reaction(void) = default;
    Ap3Reaction(const std::string & name, H5::H5File & mpo_file);

    Ap3Reaction(const Ap3Reaction & src) = default;
    ext::ap3mpo::Ap3Reaction & operator=(const ext::ap3mpo::Ap3Reaction & src) = default;
    Ap3Reaction(Ap3Reaction && src) = default;
    ext::ap3mpo::Ap3Reaction & operator=(ext::ap3mpo::Ap3Reaction && src) = default;

    std::string name;
    std::uint64_t index;
};

}  // namespace merlin

#endif  // EXT_AP3_MPO_SRC_AP3_MPO_PROPERTIES_HPP_
