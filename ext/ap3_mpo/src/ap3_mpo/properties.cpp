// Copyright 2022 quocdang1998
#include "ap3_mpo/properties.hpp"

#include <algorithm>  // std::find
#include <cstdlib>  // std::atoi
#include <cinttypes>  // PRIu64
#include <iterator>  // std::back_inserter
#include <tuple>  // std::tie
#include <unordered_set>  // std::unordered_set

#include "merlin/logger.hpp"  // WARING, FAILURE

#include "ap3_mpo/hdf5_utils.hpp"  // ap3_mpo::lowercase, ap3_mpo::check_string_in_array
                                   // ap3_mpo::get_dset, ap3_mpo::append_suffix

namespace ap3_mpo {

// --------------------------------------------------------------------------------------------------------------------
// Ap3Geometry and Ap3EnergyMesh
// --------------------------------------------------------------------------------------------------------------------

Ap3Geometry::Ap3Geometry(const std::string & name, H5::H5File & mpo_file, bool verbose) : name(name) {
    if (name.empty()) {
        FAILURE(std::invalid_argument, "Geometry cannot be empty.\n");
    }
    H5::Group geometry = mpo_file.openGroup("geometry");
    auto [geometry_names, _] = get_dset<std::string>(&geometry, "GEOMETRY_NAME");
    if (verbose) {
        std::printf("    Checking geometry name \"%s\" in MPO file: ", this->name.c_str());
    }
    this->index = check_string_in_array(this->name, geometry_names);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Geometry \"%s\" not found in MPO.\n", this->name.c_str());
    }
    this->id = append_suffix("geometry_", this->index);
    if (verbose) {
        std::printf("found (%s).\n", this->id.c_str());
    }
    H5::Group geometry_ = geometry.openGroup(this->id.c_str());
    auto [zone_name, __] = get_dset<std::string>(&geometry_, "ZONE_NAME");
    this->zone_names = std::move(zone_name);
}

Ap3EnergyMesh::Ap3EnergyMesh(const std::string & name, H5::H5File & mpo_file, bool verbose) : name(name) {
    if (name.empty()) {
        FAILURE(std::invalid_argument, "Energymesh cannot be empty.\n");
    }
    H5::Group energymesh = mpo_file.openGroup("energymesh");
    auto [energymesh_names, _] = get_dset<std::string>(&energymesh, "ENERGYMESH_NAME");
    if (verbose) {
        std::printf("    Checking energy mesh name \"%s\" in MPO file: ", this->name.c_str());
    }
    this->index = check_string_in_array(this->name, energymesh_names);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Energy mesh \"%s\" not found in MPO.\n", this->name.c_str());
    }
    this->id = append_suffix("energymesh_", this->index);
    if (verbose) {
        std::printf("found (%s).\n", this->id.c_str());
    }
    H5::Group energymesh_ = energymesh.openGroup(this->id.c_str());
    auto [energies_, __] = get_dset<float>(&energymesh_, "ENERGY");
    this->energies = std::move(energies_);
}

// --------------------------------------------------------------------------------------------------------------------
// Ap3StateParam
// --------------------------------------------------------------------------------------------------------------------

Ap3StateParam::Ap3StateParam(H5::H5File & mpo_file, bool verbose) {
    H5::Group parameters = mpo_file.openGroup("parameters");
    auto [param_names_, _] = get_dset<std::string>(&parameters, "info/PARAMNAME");
    this->param_names = std::move(param_names_);
    unsigned int nparam = this->param_names.size();
    if (verbose) {
        std::printf("    Reading parameters:\n");
    }
    for (int i_param = 0, skip = 0; i_param < nparam; i_param++) {
        std::string & param_name = this->param_names[i_param - skip];
        trim(param_name);
        // skip Time
        if (lowercase(param_name).compare("time") == 0) {
            this->param_names.erase(this->param_names.begin() + i_param - skip);
            this->excluded_index.push_back(i_param);
            skip += 1;
            continue;
        }
        // get and save values
        std::string param_dset_name = append_suffix("values/PARAM_", i_param).c_str();
        auto [param_value, n_value] = get_dset<float>(&parameters, param_dset_name.c_str());
        if (verbose) {
            std::printf("        %s:", param_name.c_str());
            for (const float & pv : param_value) std::printf(" %.2f", pv);
            std::printf("\n");
        }
        this->param_values[param_name] = std::vector<double>(param_value.begin(), param_value.end());
    }
}

// Overload add assignment
Ap3StateParam & Ap3StateParam::operator+=(Ap3StateParam & other) {
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

Ap3Isotope::Ap3Isotope(const std::string & name, H5::H5File & mpo_file, bool verbose) : name(name) {
    if (this->name.empty()) {
        FAILURE(std::invalid_argument, "Isotope cannot be empty.\n");
    }
    H5::Group isotopes_grp = mpo_file.openGroup("contents/isotopes");
    auto [isotopes, _] = get_dset<std::string>(&isotopes_grp, "ISOTOPENAME");
    if (verbose) {
        std::printf("    Checking isotope \"%s\" in MPO file: ", this->name.c_str());
    }
    this->index = check_string_in_array(this->name, isotopes);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Isotope \"%s\" not found in isotope list.\n", this->name.c_str());
    }
    if (verbose) {
        std::printf("okay.\n");
    }
}

Ap3Reaction::Ap3Reaction(const std::string & name, H5::H5File & mpo_file, bool verbose) : name(name) {
    if (this->name.empty()) {
        FAILURE(std::invalid_argument, "Reaction cannot be empty.\n");
    }
    H5::Group reactions_grp = mpo_file.openGroup("contents/reactions");
    auto [reactions, _] = get_dset<std::string>(&reactions_grp, "REACTIONAME");
    if (verbose) {
        std::printf("    Checking reaction \"%s\" in MPO file: ", this->name.c_str());
    }
    this->index = check_string_in_array(this->name, reactions);
    if (this->index == UINT64_MAX) {
        FAILURE(std::runtime_error, "Reaction \"%s\" not found in isotope list.\n", this->name.c_str());
    }
    if (verbose) {
        std::printf("okay.\n");
    }
}

}  // namespace ap3_mpo
