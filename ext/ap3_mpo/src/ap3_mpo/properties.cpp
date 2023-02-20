// Copyright 2022 quocdang1998
#include "ap3_mpo/properties.hpp"

#include <algorithm>  // std::find
#include <cstdlib>  // std::atoi
#include <cinttypes>  // PRIu64
#include <iterator>  // std::back_inserter
#include <tuple>  // std::tie
#include <unordered_set>  // std::unordered_set

#include "merlin/logger.hpp"  // WARING, FAILURE
#include "merlin/utils.hpp"  // merlin::ndim_to_contiguous_idx

#include "ap3_mpo/hdf5_utils.hpp"  // merlin::ext::ap3mpo::lowercase, merlin::ext::ap3mpo::check_string_in_array
                                   // merlin::ext::ap3mpo::get_dset, merlin::ext::ap3mpo::append_suffix

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Ap3Geometry and Ap3EnergyMesh
// --------------------------------------------------------------------------------------------------------------------

ext::ap3mpo::Ap3Geometry::Ap3Geometry(const std::string & name, H5::H5File & mpo_file) : name(name) {
    if (name.empty()) {
        FAILURE(std::invalid_argument, "Geometry cannot be empty.\n");
    }
    H5::Group geometry = mpo_file.openGroup("geometry");
    auto [geometry_names, _] = ext::ap3mpo::get_dset<std::string>(&geometry, "GEOMETRY_NAME");
    std::printf("    Checking geometry name \"%s\" in MPO file: ", this->name.c_str());
    this->index = ext::ap3mpo::check_string_in_array(this->name, geometry_names);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Geometry \"%s\" not found in MPO.\n", this->name.c_str());
    }
    this->id = ext::ap3mpo::append_suffix("geometry_", this->index);
    std::printf("found (%s).\n", this->id.c_str());
    H5::Group geometry_ = geometry.openGroup(this->id.c_str());
    auto [zone_name, __] = ext::ap3mpo::get_dset<std::string>(&geometry_, "ZONE_NAME");
    this->zone_names = std::move(zone_name);
}

ext::ap3mpo::Ap3EnergyMesh::Ap3EnergyMesh(const std::string & name, H5::H5File & mpo_file) : name(name) {
    if (name.empty()) {
        FAILURE(std::invalid_argument, "Energymesh cannot be empty.\n");
    }
    H5::Group energymesh = mpo_file.openGroup("energymesh");
    auto [energymesh_names, _] = ext::ap3mpo::get_dset<std::string>(&energymesh, "ENERGYMESH_NAME");
    std::printf("    Checking energy mesh name \"%s\" in MPO file: ", this->name.c_str());
    this->index = ext::ap3mpo::check_string_in_array(this->name, energymesh_names);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Energy mesh \"%s\" not found in MPO.\n", this->name.c_str());
    }
    this->id = ext::ap3mpo::append_suffix("energymesh_", this->index);
    std::printf("found (%s).\n", this->id.c_str());
    H5::Group energymesh_ = energymesh.openGroup(this->id.c_str());
    auto [energies_, __] = ext::ap3mpo::get_dset<float>(&energymesh_, "ENERGY");
    this->energies = std::move(energies_);
}

// --------------------------------------------------------------------------------------------------------------------
// Ap3StateParam
// --------------------------------------------------------------------------------------------------------------------

ext::ap3mpo::Ap3StateParam::Ap3StateParam(H5::H5File & mpo_file) {
    H5::Group parameters = mpo_file.openGroup("parameters");
    auto [param_names_, _] = ext::ap3mpo::get_dset<std::string>(&parameters, "info/PARAMNAME");
    this->param_names = std::move(param_names_);
    unsigned int nparam = this->param_names.size();
    std::printf("    Reading parameters:\n");
    for (int i_param = 0, skip = 0; i_param < nparam; i_param++) {
        std::string & param_name = this->param_names[i_param - skip];
        ext::ap3mpo::trim(param_name);
        // skip Time
        if (ext::ap3mpo::lowercase(param_name).compare("time") == 0) {
            this->param_names.erase(this->param_names.begin() + i_param - skip);
            this->excluded_index.push_back(i_param);
            skip += 1;
            continue;
        }
        // get and save values
        std::string param_dset_name = ext::ap3mpo::append_suffix("values/PARAM_", i_param).c_str();
        auto [param_value, n_value] = ext::ap3mpo::get_dset<float>(&parameters, param_dset_name.c_str());
        std::printf("        %s:", param_name.c_str());
        for (const float & pv : param_value) std::printf(" %.2f", pv);
        std::printf("\n");
        this->param_values[param_name] = std::vector<double>(param_value.begin(), param_value.end());
    }
}

// Overload add assignment
ext::ap3mpo::Ap3StateParam & ext::ap3mpo::Ap3StateParam::operator+=(ext::ap3mpo::Ap3StateParam & other) {
    // if this is empty pspace, copy
    if (this->param_names.empty()) {
        *this = other;
        return *this;
    }
    // check if param names are equals
    std::unordered_set<std::string> uord_set1(this->param_names.begin(), this->param_names.end());
    std::unordered_set<std::string> uord_set2(other.param_names.begin(), other.param_names.end());
    if (!(uord_set1 == uord_set2)) {
        FAILURE(std::invalid_argument, "Different param names.\n");
    }
    // merge value
    for (const std::string & pname : this->param_names) {
        std::vector<double> old_values = this->param_values[pname];
        this->param_values[pname] = std::vector<double>();
        std::set_union(old_values.begin(), old_values.end(),
                       other.param_values[pname].begin(), other.param_values[pname].end(),
                       std::back_inserter(this->param_values[pname]));
    }
    return *this;
}

// --------------------------------------------------------------------------------------------------------------------
// Ap3Isotope and Ap3Reaction
// --------------------------------------------------------------------------------------------------------------------

ext::ap3mpo::Ap3Isotope::Ap3Isotope(const std::string & name, H5::H5File & mpo_file) : name(name) {
    if (this->name.empty()) {
        FAILURE(std::invalid_argument, "Isotope cannot be empty.\n");
    }
    H5::Group isotopes_grp = mpo_file.openGroup("contents/isotopes");
    auto [isotopes, _] = ext::ap3mpo::get_dset<std::string>(&isotopes_grp, "ISOTOPENAME");
    std::printf("    Checking isotope \"%s\" in MPO file: ", this->name.c_str());
    this->index = ext::ap3mpo::check_string_in_array(this->name, isotopes);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Isotope \"%s\" not found in isotope list.\n", this->name.c_str());
    }
    std::printf("okay.\n");
}

ext::ap3mpo::Ap3Reaction::Ap3Reaction(const std::string & name, H5::H5File & mpo_file) : name(name) {
    if (this->name.empty()) {
        FAILURE(std::invalid_argument, "Reaction cannot be empty.\n");
    }
    H5::Group reactions_grp = mpo_file.openGroup("contents/reactions");
    auto [reactions, _] = ext::ap3mpo::get_dset<std::string>(&reactions_grp, "REACTIONAME");
    std::printf("    Checking reaction \"%s\" in MPO file: ", this->name.c_str());
    this->index = ext::ap3mpo::check_string_in_array(this->name, reactions);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Reaction \"%s\" not found in isotope list.\n", this->name.c_str());
    }
    std::printf("okay.\n");
}

}  // namespace merlin
