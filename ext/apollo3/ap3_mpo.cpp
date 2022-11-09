// Copyright 2022 quocdang1998
#include "ap3_mpo.hpp"

#include <algorithm>  // std::find
#include <cstdlib>  // std::atoi
#include <tuple>  // std::tie

#include "H5Cpp.h"  // H5::Group, H5::H5File, H5F_ACC_RDONLY
#include "merlin/logger.hpp"  // MESSAGE, WARING, FAILURE
#include "merlin/utils.hpp"  // merlin::ndim_to_contiguous_idx

#include "hdf5_utils.hpp"  // ls_groups

// search index of isotope and reaction in MPO file
static std::pair<std::uint64_t, std::uint64_t> search(H5::Group * root, const std::string & iso, const std::string & reac) {
    auto [isotopes, _] = get_dset<std::string>(root, "contents/isotopes/ISOTOPENAME");
    std::uint64_t i_iso = check_string_in_array(iso, isotopes);
    if (i_iso == UINT64_MAX) {
        FAILURE(std::runtime_error, "Isotope %s not found in isotope list.\n", iso.c_str());
    }
    auto [reactions, __] = get_dset<std::string>(root, "contents/reactions/REACTIONAME");
    std::uint64_t i_reac = check_string_in_array(reac, reactions);
    if (i_reac == UINT64_MAX) {
        FAILURE(std::runtime_error, "Reaction %s not found in reaction list.\n", reac.c_str());
    }
    return std::pair(i_iso, i_reac);
}

// get list of zone names
static std::vector<std::string> query_geometry(H5::Group * root) {
    H5::Group geometry = root->openGroup("geometry");
    std::vector<std::string> geometry_names = ls_groups(&geometry, "geometry_");
    if (geometry_names.size() != 1) {
        FAILURE(std::invalid_argument, "Expected 1 geometry , got %zu\n", geometry_names.size());
    }
    H5::Group geometry_0 = geometry.openGroup(geometry_names[0].c_str());
    auto [result, _] = get_dset<std::string>(&geometry_0, "ZONE_NAME");
    geometry_0.close();
    geometry.close();
    return result;
}

Ap3HomogXS::Ap3HomogXS(const std::string & filename,
                                           const std::string & isotope, const std::string & reaction,
                                           unsigned int energy_group) {
    // read metadata
    this->isotope_ = isotope;
    if (isotope.empty()) {
        FAILURE(std::invalid_argument, "Isotope cannot be empty.\n");
    }
    this->reaction_ = reaction;
    if (reaction.empty()) {
        FAILURE(std::invalid_argument, "Reaction cannot be empty.\n");
    }
    this->energy_group_ = energy_group;
    // open file
    H5::H5File mpofile(filename.c_str(), H5F_ACC_RDONLY);
    H5::Group root = mpofile.openGroup("/");
    // get index of isotope and reaction
    auto [i_iso, i_reac] = search(&root, isotope, reaction);
    // query geometry
    std::vector<std::string> zone_names = query_geometry(&root);

    // loop over each output
    H5::Group outputs = root.openGroup("output");
    std::vector<std::string> output_names = ls_groups(&outputs, "output_");
    for (const std::string & output_name : output_names) {
        // open output
        H5::Group output = outputs.openGroup(output_name.c_str());
        // check if isotope and reaction are in info/
        merlin::intvec _;
        std::vector<int> output_iso, output_reac;
        std::tie(output_iso, _) = get_dset<int>(&output, "info/ISOTOPE");
        if (std::find(output_iso.begin(), output_iso.end(), i_iso) == output_iso.end()) {
            WARNING("Isotope %s not found at output %s in MPO file %s.\n",
                    isotope.c_str(), output_name.c_str(), filename.c_str());
        }
        std::tie(output_reac, _) = get_dset<int>(&output, "info/REACTION");
        if (std::find(output_reac.begin(), output_reac.end(), i_reac) == output_reac.end()) {
            WARNING("Reaction %s not found at output %s in MPO file %s.\n",
                    reaction.c_str(), output_name.c_str(), filename.c_str());
        }
        // get addrxs
        auto [addrxs_arr, addrxs_shape] = get_dset<int>(&output, "info/ADDRXS");
        // loop over each state point
        std::vector<std::string> statept_names = ls_groups(&output, "statept_");
        for (const std::string & statept_name : statept_names) {
            // open statept
            H5::Group statept = output.openGroup(statept_name.c_str());
            // loop over each zone
            std::vector<std::string> zones = ls_groups(&output, "zone_");
            MESSAGE("Output %s Statept %s Zone_num %d.\n", output_name.c_str(), statept_name.c_str(), zones.size());
            for (const std::string & zone_id : zones) {
                // open zone
                H5::Group zone = statept.openGroup(zone_id.c_str());
                // get address of cross section
                std::vector<int> addrzx_array;
                std::tie(addrzx_array, _) = get_dset<int>(&zone, "ADDRZX");
                std::uint64_t addrzx = addrzx_array[0];
                std::uint64_t addrxs = addrxs_arr[merlin::ndim_to_contiguous_idx({i_reac, i_iso, addrzx}, addrxs_shape)];
                MESSAGE("Output %s Statept %s Zone %s Addrxs: %u.\n", output_name.c_str(), statept_name.c_str(), zone_id.c_str(), addrxs);
                // close zone
                zone.close();
            }
            // close statept
            statept.close();
        }
        // close output
        output.close();
    }
    outputs.close();

    // close file
    root.close();
    mpofile.close();
}

