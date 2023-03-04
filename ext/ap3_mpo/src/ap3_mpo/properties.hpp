// Copyright 2022 quocdang1998
#ifndef EXT_AP3_MPO_SRC_AP3_MPO_PROPERTIES_HPP_
#define EXT_AP3_MPO_SRC_AP3_MPO_PROPERTIES_HPP_

#include <cstdint>  // std::uint64_t
#include <map>  // std::map
#include <string>  // std::string
#include <vector>  // std::vector

#include "H5Cpp.h"  // H5::File

#include "ap3_mpo/declaration.hpp"  // ap3_mpo::Ap3Geometry, ap3_mpo::Ap3EnergyMesh
                                    // ap3_mpo::Ap3StateParam, ap3_mpo::Ap3Isotope

namespace ap3_mpo {

struct Ap3Geometry {
    Ap3Geometry(void) = default;
    Ap3Geometry(const std::string & name, H5::H5File & mpo_file);

    Ap3Geometry(const Ap3Geometry & src) = default;
    Ap3Geometry & operator=(const Ap3Geometry & src) = default;
    Ap3Geometry(Ap3Geometry && src) = default;
    Ap3Geometry & operator=(Ap3Geometry && src) = default;

    std::string id;
    std::uint64_t index;
    std::string name;
    std::vector<std::string> zone_names;
};

struct Ap3EnergyMesh {
    Ap3EnergyMesh(void) = default;
    Ap3EnergyMesh(const std::string & name, H5::H5File & mpo_file);

    Ap3EnergyMesh(const Ap3EnergyMesh & src) = default;
    Ap3EnergyMesh & operator=(const Ap3EnergyMesh & src) = default;
    Ap3EnergyMesh(Ap3EnergyMesh && src) = default;
    Ap3EnergyMesh & operator=(Ap3EnergyMesh && src) = default;

    std::string id;
    std::uint64_t index;
    std::string name;
    std::vector<float> energies;
};

struct Ap3StateParam {
    Ap3StateParam(void) = default;
    Ap3StateParam(H5::H5File & mpo_file);

    Ap3StateParam(const Ap3StateParam & src) = default;
    Ap3StateParam & operator=(const Ap3StateParam & src) = default;
    Ap3StateParam(Ap3StateParam && src) = default;
    Ap3StateParam & operator=(Ap3StateParam && src) = default;

    Ap3StateParam & operator+=(Ap3StateParam & other);

    std::vector<std::string> param_names;
    std::map<std::string, std::vector<double>> param_values;
    std::vector<std::uint64_t> excluded_index;
};

struct Ap3Isotope {
    Ap3Isotope(void) = default;
    Ap3Isotope(const std::string & name, H5::H5File & mpo_file);

    Ap3Isotope(const Ap3Isotope & src) = default;
    Ap3Isotope & operator=(const Ap3Isotope & src) = default;
    Ap3Isotope(Ap3Isotope && src) = default;
    Ap3Isotope & operator=(Ap3Isotope && src) = default;

    std::string name;
    std::uint64_t index;
};

struct Ap3Reaction {
    Ap3Reaction(void) = default;
    Ap3Reaction(const std::string & name, H5::H5File & mpo_file);

    Ap3Reaction(const Ap3Reaction & src) = default;
    Ap3Reaction & operator=(const Ap3Reaction & src) = default;
    Ap3Reaction(Ap3Reaction && src) = default;
    Ap3Reaction & operator=(Ap3Reaction && src) = default;

    std::string name;
    std::uint64_t index;
};

}  // namespace ap3_mpo

#endif  // EXT_AP3_MPO_SRC_AP3_MPO_PROPERTIES_HPP_
