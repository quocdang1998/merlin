// Copyright 2022 quocdang1998
#include "ap3_mpo/ap3_xs.hpp"

#include <cinttypes>  // PRIu64
#include <iterator>   // std::back_inserter

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"   // merlin::ndim_to_contiguous_idx

#include "ap3_mpo/hdf5_utils.hpp"  // ap3_mpo::ls_groups, ap3_mpo::check_string_in_array
                                   // ap3_mpo::get_dset, ap3_mpo::append_suffix
                                   // ap3_mpo::find_element

namespace ap3_mpo {

// ---------------------------------------------------------------------------------------------------------------------
// Ap3HomogXS
// ---------------------------------------------------------------------------------------------------------------------

// Constructor
Ap3HomogXS::Ap3HomogXS(const std::string & filename, const std::string & geometry_id,
                       const std::string & energy_mesh_id, const std::string & isotope, const std::string & reaction,
                       bool verbose) :
mpo_file_(filename.c_str(), H5F_ACC_RDONLY), verbose(verbose) {
    // open file and check input
    if (verbose) {
        std::printf("    Open HDF5 file \"%s\".\n", filename.c_str());
    }
    this->geometry_ = Ap3Geometry(geometry_id, this->mpo_file_, verbose);
    this->energymesh_ = Ap3EnergyMesh(energy_mesh_id, this->mpo_file_, verbose);
    this->isotope_ = Ap3Isotope(isotope, this->mpo_file_, verbose);
    this->reaction_ = Ap3Reaction(reaction, this->mpo_file_, verbose);
    this->state_param_ = Ap3StateParam(this->mpo_file_, verbose);
    // get ouput id
    H5::Group outputs = this->mpo_file_.openGroup("output");
    auto [output_ids, _shape] = get_dset<int>(&outputs, "OUPUTID");
    std::uint64_t output_idx = merlin::ndim_to_contiguous_idx({this->geometry_.index, this->energymesh_.index}, _shape);
    int output_id = output_ids[output_idx];
    std::string output_id_str = append_suffix("output_", output_id);
    if (verbose) {
        std::printf("    Got output ID: %s.\n", output_id_str.c_str());
    }
    this->output_ = outputs.openGroup(output_id_str.c_str());
}

// Merge 2 objects
Ap3HomogXS & Ap3HomogXS::operator+=(Ap3HomogXS & other) {
    // check if left and right are coherent
    if (!(this->geometry_.name.empty()) && (this->geometry_.name.compare(other.geometry_.name) != 0)) {
        FAILURE(std::invalid_argument, "Left and right of operator add must have the same geometry, got %s and %s.\n",
                this->geometry_.name.c_str(), other.geometry_.name.c_str());
    }
    if (!(this->energymesh_.name.empty()) && (this->energymesh_.name.compare(other.energymesh_.name) != 0)) {
        FAILURE(std::invalid_argument, "Left and right of operator add must have the same energymesh, got %s and %s.\n",
                this->energymesh_.name.c_str(), other.energymesh_.name.c_str());
    }
    if (!(this->isotope_.name.empty()) && (this->isotope_.name.compare(other.isotope_.name) != 0)) {
        FAILURE(std::invalid_argument, "Left and right of operator add must have the same isotope, got %s and %s.\n",
                this->isotope_.name.c_str(), other.isotope_.name.c_str());
    }
    if (!(this->reaction_.name.empty()) && (this->reaction_.name.compare(other.reaction_.name) != 0)) {
        FAILURE(std::invalid_argument, "Left and right of operator add must have the same reaction, got %s and %s.\n",
                this->reaction_.name.c_str(), other.reaction_.name.c_str());
    }
    // copy from other value
    this->geometry_ = other.geometry_;
    this->energymesh_ = other.energymesh_;
    this->isotope_ = other.isotope_;
    this->reaction_ = other.reaction_;
    // merge 2 parameter space and list of instances
    this->state_param_ += other.state_param_;
    std::list<Ap3HomogXS *> old_linked_instances = this->linked_instances_;
    this->linked_instances_.clear();
    std::set_union(old_linked_instances.begin(), old_linked_instances.end(), other.linked_instances_.begin(),
                   other.linked_instances_.end(), std::back_inserter(this->linked_instances_));
    this->linked_instances_.push_back(&other);
    return *this;
}

// Get shape of parameter space
merlin::intvec Ap3HomogXS::get_output_shape(void) {
    merlin::intvec result(this->state_param_.param_names.size() + 2);
    result[0] = this->geometry_.zone_names.size();
    for (std::uint64_t i_param = 1; i_param < result.size() - 1; i_param++) {
        result[i_param] = this->state_param_.param_values[this->state_param_.param_names[i_param - 1]].size();
    }
    result[result.size() - 1] = this->energymesh_.energies.size() - 1;
    return result;
}

// Assign Stock
void Ap3HomogXS::assign_destination_array(merlin::array::NdData & dest) {
    this->pdata_ = &dest;
    for (Ap3HomogXS * linked : this->linked_instances_) {
        linked->assign_destination_array(dest);
    }
}

// Get microscopic homogenized cross-section to stock
void Ap3HomogXS::write_to_stock(const Ap3StateParam & pspace, const std::string & xstype) {
    // check xstype
    static std::vector<std::string> allowed_xstype = {"micro", "macro", "zoneflux", "RR"};
    std::uint64_t xs_type_index = check_string_in_array(xstype, allowed_xstype);
    if (xs_type_index == UINT64_MAX) {
        FAILURE(std::invalid_argument, "Invalid argument xstype (%s).\n", xstype.c_str());
    }
    // recursively execute on each instance
    if (this->linked_instances_.size() > 0) {
        for (Ap3HomogXS * linked : this->linked_instances_) {
            linked->write_to_stock(pspace, xstype);
        }
        return;
    }
    // get address to cross section
    MESSAGE("Processing data from MPO file \"%s\"...\n", this->mpo_file_.getFileName().c_str());
    auto [addrxs, addrxs_shape] = get_dset<int>(&(this->output_), "info/ADDRXS");
    // get index of isotope and reaction
    auto [i_iso_in_geo, num_iso_in_geo] = get_dset<int>(&(this->output_), "info/ISOTOPE");
    std::uint64_t i_iso = find_element(i_iso_in_geo, this->isotope_.index);
    if (i_iso == UINT64_MAX) {
        WARNING("Isotope not found in the geometry, doing nothing.\n");
        return;
    }
    if (this->verbose) {
        std::printf("    Found isotope in geometry (index %" PRIu64 ").\n", i_iso);
    }
    auto [i_reac_in_geo, num_reac_in_geo] = get_dset<int>(&(this->output_), "info/REACTION");
    std::uint64_t i_reac = find_element(i_reac_in_geo, this->reaction_.index);
    if (i_reac == UINT64_MAX) {
        WARNING("Reaction is not considered in the geometry, doing nothing.\n");
        return;
    }
    if (this->verbose) {
        std::printf("    Found reaction in geometry (index %" PRIu64 ").\n", i_reac);
    }
    // loop on each statept
    std::vector<std::string> statepts = ls_groups(&(this->output_), "statept_");
    if (this->verbose) {
        std::printf("    Loop on each statept:");
    }
    for (const std::string & statept_str : statepts) {
        H5::Group statept = this->output_.openGroup(statept_str.c_str());
        std::uint64_t statept_idx = get_suffix(statept_str, "statept_");
        if (this->verbose) {
            std::printf(" %" PRIu64, statept_idx);
        }
        // calculate index of parameters wrt the storing array
        merlin::intvec stock_index(pspace.param_names.size() + 2);
        auto [paramvalueord, paramvalueord_shape] = get_dset<int>(&statept, "PARAMVALUEORD");
        for (int i_excluded = this->state_param_.excluded_index.size() - 1; i_excluded >= 0; i_excluded--) {
            paramvalueord.erase(paramvalueord.begin() + this->state_param_.excluded_index[i_excluded]);
        }
        for (int i_param = 0; i_param < pspace.param_names.size(); i_param++) {
            const std::string & param_name = pspace.param_names[i_param];
            std::uint64_t i_paramvalueord = find_element(this->state_param_.param_names, param_name);
            double & param_value = this->state_param_.param_values[param_name][paramvalueord[i_paramvalueord]];
            std::uint64_t i_param_in_stock = find_element(pspace.param_values.at(param_name), param_value);
            stock_index[i_param + 1] = i_param_in_stock;
        }
        // loop on each zone
        std::vector<std::string> zones = ls_groups(&statept, "zone_");
        std::vector<double> concentration(zones.size());
        std::vector<std::vector<double>> zoneflux(zones.size(),
                                                  std::vector<double>(this->energymesh_.energies.size() - 1));
        std::vector<std::vector<double>> crossection(zones.size(),
                                                     std::vector<double>(this->energymesh_.energies.size() - 1));
        for (const std::string & zone_str : zones) {
            H5::Group zone = statept.openGroup(zone_str.c_str());
            std::uint64_t zone_idx = get_suffix(zone_str, "zone_");
            // get isotope concentration
            auto [concentrations, concentrations_shape] = get_dset<float>(&zone, "CONCENTRATION");
            concentration[zone_idx] = concentrations[i_iso];
            // get zone flux
            auto [zonefluxs, zoneflux_shape] = get_dset<float>(&zone, "ZONEFLUX");
            for (int i_group = 0; i_group < zoneflux[zone_idx].size(); i_group++) {
                zoneflux[zone_idx][i_group] = zonefluxs[i_group];
            }
            // get address of zone and get cross section
            auto [addrzx_data, _] = get_dset<int>(&zone, "ADDRZX");
            std::uint64_t addrzx = addrzx_data[0];
            int crossection_index =
                addrxs[merlin::ndim_to_contiguous_idx(merlin::intvec({addrzx, i_iso, i_reac}), addrxs_shape)];
            auto [crossections, crossections_shape] = get_dset<float>(&zone, "CROSSECTION");
            for (int i_group = 0; i_group < crossection[zone_idx].size(); i_group++) {
                crossection[zone_idx][i_group] = crossections[crossection_index + i_group];
            }
            // save crossection to stock
            stock_index[0] = zone_idx;
            for (int i_group = 0; i_group < crossection[zone_idx].size(); i_group++) {
                stock_index[stock_index.size() - 1] = i_group;
                switch (xs_type_index) {
                    case 0 : {  // micro
                        this->pdata_->set(stock_index, crossection[zone_idx][i_group]);
                        break;
                    }
                    case 1 : {  // macro
                        this->pdata_->set(stock_index, concentration[zone_idx] * crossection[zone_idx][i_group]);
                        break;
                    }
                    case 2 : {  // zoneflux
                        this->pdata_->set(stock_index, zoneflux[zone_idx][i_group]);
                        break;
                    }
                    case 3 : {  // RR
                        double macro_xs = concentration[zone_idx] * crossection[zone_idx][i_group];
                        this->pdata_->set(stock_index, zoneflux[zone_idx][i_group] * macro_xs);
                        break;
                    }
                }
            }
        }
    }
    if (this->verbose) {
        std::printf("\n");
    }
}

}  // namespace ap3_mpo
